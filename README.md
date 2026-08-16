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

The branch `agent/tempflow-psnr-only-4gpu` is a standalone checkout. It does not import or modify an
`RL4ACWM-publish` worktree. The four-GPU runner forms one global six-branch TempFlow group: ranks 0
and 1 collect two branches each, ranks 2 and 3 collect one each; rewards are gathered globally,
PSNR advantages are computed once over all six branches, and LoRA gradients are summed before every
optimizer step. Only rank 0 writes checkpoints, group logs and fixed16 evaluations.

The remote recipe is intentionally PSNR-only. It uses raw future-only PSNR, branch factor 6, the
existing learning rate/noise weighting/PPO settings, and the agreed `reference_kl_beta: 0.5`.
Action reward and its tracker dependencies are not initialized.

On a machine with exactly four visible GPUs:

```bash
git clone --single-branch --branch agent/tempflow-psnr-only-4gpu \
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
