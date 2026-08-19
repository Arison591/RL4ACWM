# Upstream security notes

The experimental source checkout contains credential-related edits in documentation/runtime support.
No credential value was inspected, copied, or committed here. This repository defaults to
`WANDB_MODE=offline`; credentials, if needed later, must be injected through the environment.

Asset-path and RNG concerns are handled by adapters. No upstream patch is currently applied.

