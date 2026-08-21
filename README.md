# RL4ACWM TempFlow Video

Standalone TempFlow-GRPO training code for video AWM. The policy implementation is not vendored: set
`AWM_UPSTREAM_ROOT` to a clean, detached checkout of GE-Sim/AWM at commit
`dce69e48a952449e873a791812e506df878bc8a9`. Model/data assets are addressed through
`AWM_ASSET_ROOT`; generated artifacts stay outside Git. W&B defaults to offline mode.

The repository supports legacy total-reward normalization for parity and component-wise Action/PSNR
advantages for new training. Formal training refuses unset component variance thresholds.

The complete pre-extraction research implementation is retained under `legacy_source/` for audit and
porting reference. The original `RL4ACWM-publish` checkout no longer contains TempFlow directories,
scripts, tests, tools, or README references.

```bash
export AWM_UPSTREAM_ROOT=/path/to/RL4ACWM-upstream-clean
export AWM_ASSET_ROOT=/path/to/private/assets
python tools/audit_upstream.py
pytest
```

## PSNR-only four-GPU training

The four-GPU runner forms global TempFlow groups from rank-local branch shards. Rewards are gathered
globally, PSNR advantages are computed once over the complete group, and LoRA gradients are summed
before every optimizer step. A deterministic prefix is reused across all selected timesteps while
the old policy stays frozen; only after collection does optimization start. Only rank 0 writes
checkpoints, group logs and evaluations.

The full recipe is intentionally PSNR-only. It uses raw future-only PSNR, branch factor 6,
`learning_rate: 1e-5`, noise-aware weighting, and `reference_kl_beta: 0.5`.
Action reward and its tracker dependencies are not initialized.

On a machine with exactly four visible GPUs:

```bash
git clone --single-branch --branch agent/tempflow-overfit-signal-fix \
  https://github.com/Arison591/RL4ACWM.git RL4ACWM-tempflow-video
cd RL4ACWM-tempflow-video

export AWM_UPSTREAM_ROOT=/path/to/RL4ACWM-upstream-clean
export AWM_ASSET_ROOT=/path/to/Genie-Envisioner-V1
export TEMPFLOW_DATA_ROOT=/path/to/awm_coca_overfit16
export TEMPFLOW_OUTPUT_ROOT=/path/to/tempflow_outputs
scripts/run_psnr_only_4gpu.sh
```

Alternatively, `scripts/bootstrap_psnr_only_4gpu.sh /path/to/new/workspace` creates separate
`RL4ACWM-tempflow-video`, `RL4ACWM-upstream-clean`, and `outputs` folders without overwriting an
existing path. Model assets and datasets remain external.

The committed implementation has CPU tests for global branch sharding and metric weighting. A true
four-GPU NCCL smoke must still be run on the destination machine before committing a 224-group job;
the launcher refuses any visible GPU count other than four.

For the recommended signal-first diagnosis, choose one of the 16 condition IDs and run:

```bash
export TEMPFLOW_OVERFIT_CONDITION_ID=<condition-id>
scripts/run_psnr_signal_overfit_4gpu.sh
```

This uses timestep 2, 12 branches, KL beta 0.01, a 0.005 dB variance floor, two PPO passes per
frozen-policy group, and train-seed plus held-out-seed evaluation every 10 optimizer steps.
