#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PYTHON_BIN="${PYTHON_BIN:-/hpc2hdd/home/bohantan/.conda/envs/genie-psnr/bin/python}"
export TORCHRUN_BIN="${TORCHRUN_BIN:-/hpc2hdd/home/bohantan/.conda/envs/genie-psnr/bin/torchrun}"
export AWM_UPSTREAM_ROOT="${AWM_UPSTREAM_ROOT:-/hpc2hdd/home/bohantan/workspace/RL4ACWM-upstream-clean}"
export AWM_ASSET_ROOT="${AWM_ASSET_ROOT:-/hpc2hdd/home/bohantan/jhupload/hr_data/awm_coca_models}"
export TEMPFLOW_DATA_ROOT="${TEMPFLOW_DATA_ROOT:-/hpc2hdd/home/bohantan/jhupload/hr_data/awm_coca_overfit16}"
export TEMPFLOW_OUTPUT_ROOT="${TEMPFLOW_OUTPUT_ROOT:-/hpc2hdd/home/bohantan/jhupload/hr_data/tempflow_outputs}"
export TEMPFLOW_OVERFIT_CONDITION_ID="${TEMPFLOW_OVERFIT_CONDITION_ID:-001_task_327_episode_684757}"
export DA3_SOURCE_ROOT="${DA3_SOURCE_ROOT:-/hpc2hdd/home/bohantan/jhupload/hr_data/Depth-Anything-3/src}"
export DA3_MODEL_PATH="${DA3_MODEL_PATH:-/hpc2hdd/home/bohantan/jhupload/hr_data/DA3-BASE}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-awm-coca}"

run_tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
export WANDB_NAME="${WANDB_NAME:-da3-mono-signal-4gpu-${run_tag}}"

mkdir -p "${AWM_ASSET_ROOT}/checkpoints"
if [[ ! -e "${AWM_ASSET_ROOT}/checkpoints/gesim" ]]; then
  ln -s ../gesim "${AWM_ASSET_ROOT}/checkpoints/gesim"
fi
mkdir -p "${TEMPFLOW_OUTPUT_ROOT}"
cd "${repo_root}"
log_file="${TEMPFLOW_OUTPUT_ROOT}/da3_mono_signal_overfit_4gpu_${run_tag}.log"
pid_file="${TEMPFLOW_OUTPUT_ROOT}/da3_mono_signal_overfit_4gpu_${run_tag}.pid"
start-stop-daemon \
  --start \
  --background \
  --make-pidfile \
  --pidfile "${pid_file}" \
  --chdir "${repo_root}" \
  --output "${log_file}" \
  --startas "${repo_root}/scripts/run_da3_mono_signal_overfit_4gpu.sh" \
  -- --skip-preflight "$@"
echo "PID=$(cat "${pid_file}")"
echo "PID_FILE=${pid_file}"
echo "LOG=${log_file}"
