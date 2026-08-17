# Four-GPU PSNR-only runner

This runner implements synchronous data parallelism at the TempFlow group level without wrapping
the PEFT model in DDP. Every rank owns an identical AWM+LoRA policy and frozen reference. One global
six-branch group is sharded as `[0,4]`, `[1,5]`, `[2]`, `[3]`.

For every rollout epoch:

1. Every rank reconstructs the same deterministic AWM prefix once for a condition.
2. With the old policy frozen, each rank generates only its assigned stochastic branches at every
   selected timestep and computes three-view raw future-only PSNR locally.
3. All ranks gather `(global_branch_id, reward)`; all compute the same population mean/std and
   clipped PSNR advantages over the global six-branch group.
4. The timestep groups enter a rollout buffer. Optimizer minibatches keep their collection-time
   `old_log_prob`, so later minibatches (or a second inner epoch) can produce non-unit PPO ratios.
5. Each local loss is scaled by `1/global_branch_count`. LoRA gradients and debug term-gradient
   buffers are summed with NCCL before clipping and `optimizer.step()`.
6. Identical optimizers therefore remain synchronized. Rank 0 alone writes logs/checkpoints and runs
   fixed16 evaluation while other ranks wait at a barrier.

The global branch factor remains six; it is not silently rounded to four or eight. Output videos are
placed under rank-specific directories, preventing filesystem collisions. A missing valid branch on
any rank aborts the update instead of risking mismatched collectives.

The full configuration is `configs/psnr_only_overfit224_4gpu.yaml`: all 16 conditions, the official
0.99 timestep prefix (14 legal transitions), repeated rollout epochs, Action disabled, raw
future-only PSNR, learning rate 1e-5, a 0.005 dB variance floor, and KL beta 0.5.

The first experiment to run is `configs/psnr_signal_overfit_4gpu.yaml`. It deliberately fixes one
condition and timestep 2, raises the global branch factor to 12, lowers KL beta to 0.01, and performs
two optimizer passes over each frozen-policy group. This turns PPO clipping into an observable code
path and separates “the gradient estimator cannot learn” from “the full sampling recipe is too
noisy.” It also logs numeric parameter delta norms, consecutive-gradient cosine, and per-seed eval
statistics for the training and held-out seeds.

This host has one GPU, so only CPU/static tests are claimed locally. Before the full destination run,
invoke the same launcher with `--max-optimizer-steps 2` and verify global rollout-group rows,
`world_size=4`, the expected branch sharding, nonzero policy gradient and parameter delta, a
non-unit ratio on the second inner pass, and matching optimizer step/policy version on all ranks.
