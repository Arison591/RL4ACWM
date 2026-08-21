#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${TEMPFLOW_CONFIG:-${repo_root}/configs/psnr_only_overfit224_4gpu.yaml}"
expected_upstream_commit="dce69e48a952449e873a791812e506df878bc8a9"

for variable in AWM_UPSTREAM_ROOT AWM_ASSET_ROOT TEMPFLOW_DATA_ROOT TEMPFLOW_OUTPUT_ROOT; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing required environment variable: ${variable}" >&2
    exit 2
  fi
done

if [[ ! -d "${AWM_UPSTREAM_ROOT}/.git" ]]; then
  echo "AWM_UPSTREAM_ROOT is not a Git checkout: ${AWM_UPSTREAM_ROOT}" >&2
  exit 2
fi
actual_commit="$(git -C "${AWM_UPSTREAM_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${expected_upstream_commit}" ]]; then
  echo "AWM upstream commit mismatch: expected ${expected_upstream_commit}, got ${actual_commit}" >&2
  exit 2
fi
if [[ -n "$(git -C "${AWM_UPSTREAM_ROOT}" status --porcelain=v1 -uall)" ]]; then
  echo "AWM upstream must be clean and read-only" >&2
  exit 2
fi
if [[ ! -d "${TEMPFLOW_DATA_ROOT}/prep" ]]; then
  echo "Missing prepared conditions: ${TEMPFLOW_DATA_ROOT}/prep" >&2
  exit 2
fi
if [[ ! -f "${AWM_ASSET_ROOT}/checkpoints/gesim/ge_sim_cosmos_v0.1.safetensors" ]]; then
  echo "Missing GE-Sim checkpoint below AWM_ASSET_ROOT=${AWM_ASSET_ROOT}" >&2
  exit 2
fi

visible_gpu_count="$(${PYTHON_BIN:-python} - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "${visible_gpu_count}" != 4 ]]; then
  echo "Exactly four visible CUDA devices are required; found ${visible_gpu_count}" >&2
  exit 2
fi

mkdir -p "${TEMPFLOW_OUTPUT_ROOT}"
export PYTHONPATH="${repo_root}/legacy_source:${AWM_UPSTREAM_ROOT}:${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TEMPFLOW_STANDALONE_ROOT="${repo_root}"

exec "${TORCHRUN_BIN:-torchrun}" \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  -m experiments.tempflow_video.run \
  --config "${config_path}" \
  "$@"
