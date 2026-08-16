# Migration inventory

Source baseline: branch `exp/tempflow-video-overfit16`, HEAD `dce69e48a952449e873a791812e506df878bc8a9`.
The source had six modified tracked files and 45 untracked files before this task. Its exact status and
binary diffs were captured privately outside both repositories.

Migrated algorithm concepts: EDM/RF coordinate mapping, stochastic and deterministic transitions,
branch identity/invariants, population group standardization, PPO/GRPO surrogate, equal-variance
reference KL, noise-aware weights, reference freeze, optimizer/checkpoint state. AWM model code and
assets are not copied. Action reward remains an upstream component behind `LegacyRewardAdapter`.

The modified tracker/runtime files are not migrated. Asset paths and RNG state are isolated through
new adapters. Credential-related changes are excluded.

