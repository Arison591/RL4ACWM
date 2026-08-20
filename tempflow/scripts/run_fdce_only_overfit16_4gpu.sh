#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for variable in AWM_UPSTREAM_ROOT AWM_ASSET_ROOT TEMPFLOW_PREP_ROOT TEMPFLOW_GT_ROOT TEMPFLOW_OUTPUT_ROOT; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing required environment variable: ${variable}" >&2
    exit 2
  fi
done

if [[ ! -d "${TEMPFLOW_PREP_ROOT}" ]]; then
  echo "Missing prepared conditions: ${TEMPFLOW_PREP_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${TEMPFLOW_GT_ROOT}/samples" ]]; then
  echo "Missing selected GT samples: ${TEMPFLOW_GT_ROOT}/samples" >&2
  exit 2
fi
if [[ ! -f "${AWM_ASSET_ROOT}/checkpoints/sam3.pt" ]]; then
  echo "Missing SAM3 checkpoint below AWM_ASSET_ROOT=${AWM_ASSET_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${AWM_ASSET_ROOT}/checkpoints/cowtracker/cowtracker_model.pth" ]]; then
  echo "Missing CoWTracker checkpoint below AWM_ASSET_ROOT=${AWM_ASSET_ROOT}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
visible_gpu_count="$(${PYTHON_BIN:-python} - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "${visible_gpu_count}" != 4 ]]; then
  echo "Expected exactly four visible GPUs; found ${visible_gpu_count}" >&2
  exit 2
fi

mkdir -p "${TEMPFLOW_OUTPUT_ROOT}"
export PYTHONPATH="${repo_root}/legacy_source:${AWM_UPSTREAM_ROOT}:${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TEMPFLOW_STANDALONE_ROOT="${repo_root}"
export AWM_MODEL_ROOT="${AWM_MODEL_ROOT:-${AWM_ASSET_ROOT}/awm_coca_models}"
export WANDB_MODE="${WANDB_MODE:-online}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

exec "${TORCHRUN_BIN:-torchrun}" --standalone --nnodes=1 --nproc-per-node=4 \
  -m experiments.tempflow_video.run \
  --config "${repo_root}/configs/fdce_only_overfit16_4gpu.yaml" "$@"
