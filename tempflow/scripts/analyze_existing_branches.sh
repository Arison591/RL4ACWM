#!/usr/bin/env bash
set -euo pipefail
: "${TEMPFLOW_EXISTING_RUN:?point TEMPFLOW_EXISTING_RUN at an existing run; no videos are generated}"
python tools/calibrate_component_std.py "$TEMPFLOW_EXISTING_RUN" --output artifacts_manifest/local_component_std.json
python tools/compare_old_new_reward_fusion.py "$TEMPFLOW_EXISTING_RUN" --psnr-threshold 0.0002 --action-threshold 0.0001 --output artifacts_manifest/local_fusion_comparison.json

