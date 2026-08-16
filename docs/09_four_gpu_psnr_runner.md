# Four-GPU PSNR-only runner

This runner implements synchronous data parallelism at the TempFlow group level without wrapping
the PEFT model in DDP. Every rank owns an identical AWM+LoRA policy and frozen reference. One global
six-branch group is sharded as `[0,4]`, `[1,5]`, `[2]`, `[3]`.

For every optimizer update:

1. Every rank reconstructs the same deterministic AWM prefix for one condition/timestep.
2. Each rank generates only its assigned stochastic branches and computes three-view raw
   future-only PSNR locally.
3. All ranks gather `(global_branch_id, reward)`; all compute the same population mean/std and
   clipped PSNR advantages over the global six-branch group.
4. Each local loss is scaled by `1/global_branch_count`. LoRA gradients and debug term-gradient
   buffers are summed with NCCL before clipping and `optimizer.step()`.
5. Identical optimizers therefore remain synchronized. Rank 0 alone writes logs/checkpoints and runs
   fixed16 evaluation while other ranks wait at a barrier.

The global branch factor remains six; it is not silently rounded to four or eight. Output videos are
placed under rank-specific directories, preventing filesystem collisions. A missing valid branch on
any rank aborts the update instead of risking mismatched collectives.

The supplied remote configuration is `configs/psnr_only_overfit224_4gpu.yaml`: 16 conditions × 14
legal timesteps, 224 updates, evaluations/checkpoints at 112 and 224, Action disabled, raw
future-only PSNR, and KL beta 0.5. No model weights, data, videos, credentials or machine-specific
absolute paths are committed.

This host has one GPU, so only CPU/static tests are claimed locally. Before the full destination run,
invoke the same launcher with `--max-optimizer-steps 1` and verify one global `groups.jsonl` row,
`world_size=4`, `branches_per_rank=[2,2,1,1]`, nonzero PSNR gradient, 448 changed LoRA tensors and
matching optimizer step/policy version on all ranks.
