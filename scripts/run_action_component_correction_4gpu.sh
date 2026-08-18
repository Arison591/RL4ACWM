#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for variable in AWM_UPSTREAM_ROOT AWM_ASSET_ROOT TEMPFLOW_DATA_ROOT TEMPFLOW_OUTPUT_ROOT ACTION_REWARD_NOISE_FLOOR_FILE; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing required environment variable: ${variable}" >&2
    exit 2
  fi
done
if [[ ! -f "${ACTION_REWARD_NOISE_FLOOR_FILE}" ]]; then
  echo "Missing measured evaluator noise-floor report: ${ACTION_REWARD_NOISE_FLOOR_FILE}" >&2
  exit 2
fi
if [[ -n "$(git -C "${AWM_UPSTREAM_ROOT}" status --porcelain=v1 -uall)" ]]; then
  echo "AWM reward checkout must be committed and clean" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
visible_gpu_count="$(${PYTHON_BIN:-python} - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "${visible_gpu_count}" != 4 ]]; then
  echo "Expected exactly GPUs 4-7 to be visible; found ${visible_gpu_count} devices" >&2
  exit 2
fi
export PYTHONPATH="${repo_root}/src:${repo_root}/legacy_source:${AWM_UPSTREAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

exec "${TORCHRUN_BIN:-torchrun}" --standalone --nnodes=1 --nproc-per-node=4 \
  -m experiments.tempflow_video.run \
  --config "${repo_root}/configs/action_component_correction_4gpu.yaml" "$@"
