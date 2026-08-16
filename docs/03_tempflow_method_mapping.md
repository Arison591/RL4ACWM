# TempFlow method mapping

| Method concept | Standalone module |
|---|---|
| RF↔GE-Sim EDM coordinates, SDE mean/variance/log-prob | `core/transitions.py` |
| deterministic prefix/suffix and branch identity | `core/branching.py` + adapters |
| population group advantage | `core/group_advantage.py` |
| PPO/GRPO surrogate | `core/policy_objective.py` |
| closed-form reference KL/freeze | `core/reference_kl.py` |
| checkpoint/resume and RNG | `runtime/checkpoint.py`, `runtime/rng_isolation.py` |
| AWM policy/scheduler/condition | `adapters/gesim_*`, `condition_adapter.py` |
| existing Action/joint reward | `adapters/legacy_reward_adapter.py` |

The source implementation collects one stochastic action at a branch timestep, uses a shared
deterministic prefix, independent branch noise, and a deterministic suffix. Old/new log-probabilities
score the collected next latent under equal transition variance. A group is bound to condition,
initial seed, branch timestep, reward configuration, and policy version, and may be consumed once.

