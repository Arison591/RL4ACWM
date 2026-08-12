#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 正常情况下只需要确认这三个目录。均可在命令前用同名环境变量覆盖。
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/dataset}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/awm_coca_remote}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
ROLLOUT_RETENTION="${ROLLOUT_RETENTION:-videos}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-genie-psnr}"
export WANDB_VIDEO_EVERY="${WANDB_VIDEO_EVERY:-50}"
export WANDB_VIDEO_SAMPLES="${WANDB_VIDEO_SAMPLES:-1}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-15}"

# 兼容旧环境变量；新接口使用 ROLLOUT_RETENTION=all|videos|none。
if [[ -n "${KEEP_ROLLOUTS:-}" ]]; then
  if [[ "${KEEP_ROLLOUTS}" == "1" ]]; then
    ROLLOUT_RETENTION="all"
  elif [[ "${KEEP_ROLLOUTS}" == "0" ]]; then
    ROLLOUT_RETENTION="none"
  else
    echo "[ERROR] KEEP_ROLLOUTS 只接受 0 或 1" >&2
    exit 2
  fi
fi
case "${ROLLOUT_RETENTION}" in
  all|videos|none) ;;
  *) echo "[ERROR] ROLLOUT_RETENTION 只接受 all、videos 或 none" >&2; exit 2 ;;
esac

if [[ "${WANDB_MODE}" == "online" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[WARN] 未设置 WANDB_API_KEY；除非本机已执行 wandb login，否则会自动退回本地日志。" >&2
  fi
  if [[ -z "${WANDB_ENTITY:-}" ]]; then
    echo "[WARN] 未设置 WANDB_ENTITY；run 可能进入对方私人空间，我们可能无权查看。" >&2
  fi
fi

resolve_from_repo() {
  local value="$1"
  if [[ "${value}" == /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${REPO_ROOT}/${value}"
  fi
}

DATA_DIR="$(resolve_from_repo "${DATA_DIR}")"
MODEL_DIR="$(resolve_from_repo "${MODEL_DIR}")"
OUTPUT_DIR="$(resolve_from_repo "${OUTPUT_DIR}")"

find_prep_root() {
  local candidate
  if [[ -n "${PREP_DIR:-}" ]]; then
    candidates=("$(resolve_from_repo "${PREP_DIR}")")
  else
    candidates=("${DATA_DIR}/prep" "${DATA_DIR}/output/prep" "${DATA_DIR}")
  fi
  for candidate in "${candidates[@]}"; do
    [[ -d "${candidate}" ]] || continue
    if [[ -n "$(find "${candidate}" -mindepth 2 -maxdepth 2 -name actions.npy -type f -print -quit)" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

find_gt_root() {
  local candidate
  if [[ -n "${GT_DIR:-}" ]]; then
    candidates=("$(resolve_from_repo "${GT_DIR}")")
  else
    candidates=(
      "${DATA_DIR}/selected_samples/samples"
      "${DATA_DIR}/gt"
      "${DATA_DIR}/data/agibotworld_beta/selected_samples/samples"
      "${DATA_DIR}"
    )
  fi
  for candidate in "${candidates[@]}"; do
    [[ -d "${candidate}" ]] || continue
    if [[ -n "$(find "${candidate}" -mindepth 2 -maxdepth 2 -name head_29_frames.mp4 -type f -print -quit)" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

PREP_ROOT="$(find_prep_root)" || {
  echo "[ERROR] 无法在 DATA_DIR=${DATA_DIR} 下找到 <condition_id>/actions.npy" >&2
  echo "[ERROR] 推荐结构: dataset/prep/<condition_id>/actions.npy" >&2
  exit 2
}
GT_ROOT="$(find_gt_root)" || {
  echo "[ERROR] 无法在 DATA_DIR=${DATA_DIR} 下找到 <condition_id>/head_29_frames.mp4" >&2
  echo "[ERROR] 推荐结构: dataset/selected_samples/samples/<condition_id>/*_29_frames.mp4" >&2
  exit 2
}

mkdir -p "${OUTPUT_DIR}/logs"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
CONSOLE_LOG="${OUTPUT_DIR}/logs/train_${RUN_STAMP}.log"
METADATA_FILE="${OUTPUT_DIR}/logs/run_${RUN_STAMP}.txt"
EFFECTIVE_CONFIG="${OUTPUT_DIR}/logs/effective_config_${RUN_STAMP}.yaml"

exec > >(tee -a "${CONSOLE_LOG}") 2>&1

on_exit() {
  local code=$?
  if [[ ${code} -eq 0 ]]; then
    echo "[DONE] 训练正常结束"
  else
    echo "[ERROR] 训练异常退出，exit_code=${code}" >&2
  fi
  echo "[INFO] 控制台日志: ${CONSOLE_LOG}"
  echo "[INFO] JSONL 指标: ${OUTPUT_DIR}/metrics"
  echo "[INFO] Checkpoint: ${OUTPUT_DIR}/checkpoints"
}
trap on_exit EXIT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_ASYNC_ERROR_HANDLING=1

{
  echo "timestamp=${RUN_STAMP}"
  echo "repo_root=${REPO_ROOT}"
  echo "git_branch=$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || true)"
  echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
  echo "data_dir=${DATA_DIR}"
  echo "prep_root=${PREP_ROOT}"
  echo "gt_root=${GT_ROOT}"
  echo "model_dir=${MODEL_DIR}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "nproc_per_node=${NPROC_PER_NODE}"
  echo "rollout_retention=${ROLLOUT_RETENTION}"
  echo "resume_checkpoint=${RESUME_CHECKPOINT}"
  echo "wandb_mode=${WANDB_MODE}"
  echo "wandb_project=${WANDB_PROJECT}"
  echo "wandb_entity=${WANDB_ENTITY:-}"
  echo "wandb_video_every=${WANDB_VIDEO_EVERY}"
  echo "wandb_video_samples=${WANDB_VIDEO_SAMPLES}"
  "${PYTHON_BIN}" --version
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
} | tee "${METADATA_FILE}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/check_remote_env.py" \
  --model-dir "${MODEL_DIR}" --require-gpus "${NPROC_PER_NODE}"

COMMON_ARGS=(
  --prep-root "${PREP_ROOT}"
  --gt-root "${GT_ROOT}"
  --checkpoint-root "${MODEL_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --rollout-retention "${ROLLOUT_RETENTION}"
)

if [[ "${ROLLOUT_RETENTION}" == "videos" ]]; then
  echo "[INFO] 仅保留三视角 MP4、reward、credit 和元数据；训练中间张量会在消费后删除。"
elif [[ "${ROLLOUT_RETENTION}" == "all" ]]; then
  echo "[WARN] ROLLOUT_RETENTION=all：视频和所有 latent 都会保留，请监控磁盘。"
else
  echo "[INFO] ROLLOUT_RETENTION=none：rollout 指标写入 JSONL 后删除整个 group。"
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  COMMON_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi

PYTHON_BIN="${PYTHON_BIN}" "${SCRIPT_DIR}/run_awm_coca.sh" preflight "${COMMON_ARGS[@]}"

"${PYTHON_BIN}" -m experiments.awm_coca.run_train \
  --config "${REPO_ROOT}/configs/awm_coca_train.yaml" \
  --print-effective-config \
  --effective-config-output "${EFFECTIVE_CONFIG}" \
  "${COMMON_ARGS[@]}"

echo "[INFO] 单机四卡训练开始：global group=16，每卡 4 条 rollout"
PYTHON_BIN="${PYTHON_BIN}" NPROC_PER_NODE="${NPROC_PER_NODE}" \
  "${SCRIPT_DIR}/run_awm_coca.sh" train4 "${COMMON_ARGS[@]}" "$@"
