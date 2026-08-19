# Action RL Stability Combined Revision

This branch is the single checkout for the action-following and PSNR TempFlow
experiments. The AWM-CoCA source remains at the repository root; the complete
standalone TempFlow project is under `tempflow/`.

Included revisions:

- AWM action reward audit, arm-specific command matching, detector-state reset,
  and audited fixed-condition replacements (`105 -> 150`, `144 -> 141`).
- TempFlow PSNR-only signal fixes and four-GPU runner (`6eb0a8f`), followed by
  raw-command action advantage, command gating, four-group accumulation,
  multi-timestep scheduling, SAM3 startup synchronization, and W&B logging.
- Pure action-following overfit16 configuration:
  `tempflow/configs/action_following_overfit16_4gpu.yaml`.

The TempFlow runner expects the AWM source root through `AWM_UPSTREAM_ROOT` and
uses the same external model/data environment as the AWM-CoCA scripts. The
training code keeps command reward as the primary signal; FDCE and IoU are
diagnostics in the pure action-following configuration.
