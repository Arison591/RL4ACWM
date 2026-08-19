# Next formal run plan

The standalone real PSNR-only smoke gate has passed. The approved PSNR-only configuration keeps the
existing TempFlow learning rate, KL beta, branch factor, noise weighting and PPO logic, and changes
only the objective weights to `lambda_action=0.0` and `lambda_psnr=1.0`.

The intended full schedule remains fixed 16 conditions × legal timesteps 0..13 = 224 groups, one
epoch, fixed order and seed 123456, with fixed16 evaluations at Base, update 112 and update 224. No
parameter sweep or automatic extension is permitted.

## Deliberately shortened signal run (2026-08-16)

At the user's request, the approved run was stopped after three completed optimizer updates instead
of occupying the shared GPU for the full epoch. All three groups used condition
`001_task_327_episode_684757`, successive legal timesteps 0, 1 and 2, branch factor 6 and raw
future-only PSNR dB. Action reward was disabled.

| update | timestep | PSNR group std (dB) | PSNR policy grad norm | weighted KL | changed LoRA tensors |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.245118 | 2.23631e-5 | 0 | 448 |
| 2 | 1 | 0.163719 | 3.79524e-5 | 5.67516e-11 | 448 |
| 3 | 2 | 0.213700 | 6.34052e-5 | 1.46747e-10 | 448 |

No group was skipped. Each PSNR advantage had population std approximately 1 before clipping, every
update had a nonzero PSNR gradient, and all 448 trainable LoRA tensors changed. The process was
interrupted only after update 3 had been durably appended to `groups.jsonl`; it had begun the forward
pass for update 4, so no fourth update exists. The GPU allocation was released.

The Base fixed16 evaluation, generated before this shortened run, was 22.186473 dB future-only
balanced PSNR (mean-view 23.287778, worst-view 20.534516; head 23.736849, left wrist 24.301961,
right wrist 21.824523). Because the run stopped at update 3, the scheduled update-112 and update-224
evaluations were not run, and this short signal is not evidence that fixed16 PSNR improved.

If a full run is later approved when the shared GPU is available, use only
`configs/psnr_only_overfit224.yaml`; do not alter its optimization parameters.
