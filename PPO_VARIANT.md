# PPO Variant: Scalar Mean Baseline

This tree defaults to the original scalar-mean PPO control. Gaussian element
log probabilities are averaged across the full latent before one transition
ratio is computed. The token implementation remains available in the shared
code for compatibility, but the formal signal config selects
`ppo_ratio_mode: scalar_mean`.

The scalar learning-rate comparison uses the released TempFlow-GRPO base value
of `3e-4`; the AWM reference KL coefficient remains `0.5` for a one-variable
comparison against the previous `1e-5` run.
