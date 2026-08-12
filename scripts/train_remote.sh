#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 正常情况下只需要确认模型和输出目录。数据会自动发现，也可用 DATA_DIR 覆盖。
# 目标训练机的数据盘与 workspace 分离，因此优先检查 hr_data；其他机器仍回退到仓库 dataset/。
DATA_DIR="${DATA_DIR:-}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-/hpc2hdd/home/bohantan/jhupload/hr_data}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
OUTPUT_DIR_WAS_SET=0
[[ -n "${OUTPUT_DIR}" ]] && OUTPUT_DIR_WAS_SET=1

PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
ROLLOUT_RETENTION="${ROLLOUT_RETENTION:-videos}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-awm-coca}"
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

MODEL_DIR="$(resolve_from_repo "${MODEL_DIR}")"

find_prep_root_in() {
  local data_root="$1"
  local candidate
  local candidates=(
    "${data_root}/prep"
    "${data_root}/dataset/prep"
    "${data_root}/output/prep"
    "${data_root}/dataset/output/prep"
    "${data_root}"
  )
  for candidate in "${candidates[@]}"; do
    [[ -d "${candidate}" ]] || continue
    if [[ -n "$(find "${candidate}" -mindepth 2 -maxdepth 2 -name actions.npy -type f -print -quit)" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

find_gt_root_in() {
  local data_root="$1"
  local candidate
  local candidates=(
    "${data_root}/selected_samples/samples"
    "${data_root}/dataset/selected_samples/samples"
    "${data_root}/gt"
    "${data_root}/dataset/gt"
    "${data_root}/data/agibotworld_beta/selected_samples/samples"
    "${data_root}/dataset/data/agibotworld_beta/selected_samples/samples"
    "${data_root}"
  )
  for candidate in "${candidates[@]}"; do
    [[ -d "${candidate}" ]] || continue
    if [[ -n "$(find "${candidate}" -mindepth 2 -maxdepth 2 -name head_29_frames.mp4 -type f -print -quit)" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

if [[ -n "${PREP_DIR:-}" || -n "${GT_DIR:-}" ]]; then
  DATA_DIR="${DATA_DIR:-${REMOTE_DATA_DIR}}"
  DATA_DIR="$(resolve_from_repo "${DATA_DIR}")"
  if [[ -n "${PREP_DIR:-}" ]]; then
    PREP_ROOT="$(resolve_from_repo "${PREP_DIR}")"
  else
    PREP_ROOT="$(find_prep_root_in "${DATA_DIR}")" || true
  fi
  if [[ -n "${GT_DIR:-}" ]]; then
    GT_ROOT="$(resolve_from_repo "${GT_DIR}")"
  else
    GT_ROOT="$(find_gt_root_in "${DATA_DIR}")" || true
  fi
else
  if [[ -n "${DATA_DIR}" ]]; then
    DATA_CANDIDATES=("$(resolve_from_repo "${DATA_DIR}")")
  else
    DATA_CANDIDATES=("${REMOTE_DATA_DIR}" "${REPO_ROOT}/dataset")
  fi

  DATA_DIR=""
  PREP_ROOT=""
  GT_ROOT=""
  for candidate in "${DATA_CANDIDATES[@]}"; do
    candidate="$(resolve_from_repo "${candidate}")"
    if prep_candidate="$(find_prep_root_in "${candidate}")" \
      && gt_candidate="$(find_gt_root_in "${candidate}")"; then
      DATA_DIR="${candidate}"
      PREP_ROOT="${prep_candidate}"
      GT_ROOT="${gt_candidate}"
      break
    fi
  done
fi

# 外部数据盘存在时，checkpoint、rollout 和日志也默认写到数据盘，避免占满 workspace。
# 新训练每次创建独立 run 目录，防止上一次失败留下的 rollout 与 policy/seed 重名。
# 断点续训必须显式传入原 OUTPUT_DIR 和 RESUME_CHECKPOINT。
if [[ -z "${OUTPUT_DIR}" ]]; then
  if [[ -n "${DATA_DIR}" && "${DATA_DIR}" != "${REPO_ROOT}"/* ]]; then
    OUTPUT_ROOT="${DATA_DIR}/awm_coca_outputs"
  else
    OUTPUT_ROOT="${REPO_ROOT}/outputs/awm_coca_remote"
  fi
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)_$$"
  OUTPUT_DIR="${OUTPUT_ROOT}/runs/${RUN_STAMP}"
else
  OUTPUT_DIR="$(resolve_from_repo "${OUTPUT_DIR}")"
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)_$$"
fi

if [[ -n "${RESUME_CHECKPOINT}" && "${OUTPUT_DIR_WAS_SET}" -ne 1 ]]; then
  echo "[ERROR] 断点续训必须同时设置 OUTPUT_DIR 和 RESUME_CHECKPOINT。" >&2
  exit 2
fi

if [[ ! -d "${PREP_ROOT}" ]] || [[ -z "$(find "${PREP_ROOT}" -mindepth 2 -maxdepth 2 -name actions.npy -type f -print -quit)" ]]; then
  echo "[ERROR] 找不到训练 condition：<condition_id>/actions.npy" >&2
  echo "[ERROR] 已检查 DATA_DIR=${DATA_DIR:-<自动发现失败>}" >&2
  echo "[ERROR] 可执行 DATA_DIR=/实际数据目录 bash scripts/train_remote.sh 显式指定。" >&2
  exit 2
fi
if [[ ! -d "${GT_ROOT}" ]] || [[ -z "$(find "${GT_ROOT}" -mindepth 2 -maxdepth 2 -name head_29_frames.mp4 -type f -print -quit)" ]]; then
  echo "[ERROR] 找不到奖励 GT：<condition_id>/head_29_frames.mp4" >&2
  echo "[ERROR] 已检查 DATA_DIR=${DATA_DIR:-<自动发现失败>}" >&2
  echo "[ERROR] 可执行 GT_DIR=/实际GT目录 bash scripts/train_remote.sh 显式指定。" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}/logs"
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
