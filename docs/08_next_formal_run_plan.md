# Next formal run plan

There is exactly one proposed run after explicit approval and after full AWM smoke parity passes:

- fixed 16 overfit/debug conditions, fixed order and seed 123456;
- complete condition × legal timestep schedule (0..13), branch factor 6;
- raw future-only PSNR and unchanged Action, independently standardized/clipped;
- `lambda_action=lambda_psnr=0.5`, PSNR std gate 0.0002 dB, Action gate 0.0001;
- existing learning rate, noise weighting and KL beta; no sweep;
- fixed16 evaluation at the existing seed/protocol, reporting full and future PSNR;
- stop at the configured 100 updates and checkpoints, with no automatic extension.

The command is gated and intentionally not runnable until the standalone real smoke is cleared:
`APPROVE_FORMAL_OVERFIT=1 scripts/run_overfit16.sh`.

