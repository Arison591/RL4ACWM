# RL4ACWM TempFlow Video

Standalone TempFlow-GRPO training code for video AWM. The policy implementation is not vendored: set
`AWM_UPSTREAM_ROOT` to a clean, detached checkout of GE-Sim/AWM at commit
`dce69e48a952449e873a791812e506df878bc8a9`. Model/data assets are addressed through
`AWM_ASSET_ROOT`; generated artifacts stay outside Git. W&B defaults to offline mode.

The repository supports legacy total-reward normalization for parity and component-wise Action/PSNR
advantages for new training. Formal training refuses unset component variance thresholds.

```bash
export AWM_UPSTREAM_ROOT=/path/to/RL4ACWM-upstream-clean
export AWM_ASSET_ROOT=/path/to/private/assets
python tools/audit_upstream.py
pytest
```

