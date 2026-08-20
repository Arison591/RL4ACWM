#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for variable in TEMPFLOW_OVERFIT_CONDITION_ID DA3_SOURCE_ROOT DA3_MODEL_PATH; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing required environment variable: ${variable}" >&2
    exit 2
  fi
done
if [[ ! -f "${DA3_SOURCE_ROOT}/depth_anything_3/api.py" ]]; then
  echo "Invalid DA3_SOURCE_ROOT: ${DA3_SOURCE_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${DA3_MODEL_PATH}" ]]; then
  echo "Invalid DA3_MODEL_PATH: ${DA3_MODEL_PATH}" >&2
  exit 2
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TEMPFLOW_CONFIG="${repo_root}/configs/da3_mono_signal_smoke_4gpu.yaml"
exec "${repo_root}/scripts/run_psnr_only_4gpu.sh" "$@"
