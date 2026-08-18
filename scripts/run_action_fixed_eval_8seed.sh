#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:?usage: run_action_fixed_eval_8seed.sh <checkpoint-dir> [output-dir]}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${2:-${TEMPFLOW_OUTPUT_ROOT:?set TEMPFLOW_OUTPUT_ROOT}/action_component_correction_4gpu/fixed_eval/$(date -u +%Y%m%dT%H%M%SZ)}"

for variable in AWM_UPSTREAM_ROOT AWM_ASSET_ROOT TEMPFLOW_DATA_ROOT; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing required environment variable: ${variable}" >&2
    exit 2
  fi
done
if [[ ! -f "${checkpoint}/COMPLETE" ]]; then
  echo "Checkpoint is not complete: ${checkpoint}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export AWM_MODEL_ROOT="${AWM_MODEL_ROOT:-${AWM_ASSET_ROOT}/awm_coca_models}"
export PYTHONPATH="${repo_root}/src:${repo_root}/legacy_source:${AWM_UPSTREAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TEMPFLOW_STANDALONE_ROOT="${repo_root}"

exec "${PYTHON_BIN:-python}" "${repo_root}/scripts/action_paired_eval.py" \
  --config "${repo_root}/configs/action_component_correction_4gpu.yaml" \
  --checkpoint "${checkpoint}" \
  --output-dir "${output_dir}" \
  --seeds 1001 1002 1003 1004 1005 1006 1007 1008 \
  --conditions 16
