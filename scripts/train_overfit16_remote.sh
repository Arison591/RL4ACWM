#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/hpc2hdd/home/bohantan/jhupload/hr_data}"
SUBSET_ROOT="${OVERFIT16_DATA_DIR:-${DATA_ROOT}/awm_coca_overfit16}"
PREP_ROOT="${PREP_DIR:-${SUBSET_ROOT}/prep}"
GT_ROOT="${GT_DIR:-${SUBSET_ROOT}/selected_samples/samples}"
IDS_FILE="${OVERFIT16_IDS_FILE:-${REPO_ROOT}/configs/awm_coca_overfit16_ids.txt}"
MODEL_DIR="${MODEL_DIR:-${DATA_ROOT}/awm_coca_models}"

GROUP_SIZE="${GROUP_SIZE:-16}"
MAX_OPTIMIZER_STEPS="${MAX_OPTIMIZER_STEPS:-1000}"
EVAL_EVERY_GROUP_STEPS="${EVAL_EVERY_GROUP_STEPS:-10}"
EVAL_SEEDS_PER_CONDITION="${EVAL_SEEDS_PER_CONDITION:-8}"
EVAL_ROLLOUT_BATCH_SIZE="${EVAL_ROLLOUT_BATCH_SIZE:-2}"
EVAL_SEED="${EVAL_SEED:-12345678}"
ROLLOUT_RETENTION="${ROLLOUT_RETENTION:-videos}"

fail() {
  echo "[ERROR] $*" >&2
  exit 2
}

visible_gpu_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
    awk -F',' '{ print NF }' <<<"${CUDA_VISIBLE_DEVICES}"
    return
  fi
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  nvidia-smi -L | awk 'END { print NR }'
}

NPROC_PER_NODE="${NPROC_PER_NODE:-}"
if [[ -z "${NPROC_PER_NODE}" ]]; then
  NPROC_PER_NODE="$(visible_gpu_count)" || fail "无法检测 GPU；请设置 NPROC_PER_NODE=4 或 8"
fi
[[ "${NPROC_PER_NODE}" == "4" || "${NPROC_PER_NODE}" == "8" ]] \
  || fail "小型 overfit 脚本仅支持 4 或 8 卡，收到 NPROC_PER_NODE=${NPROC_PER_NODE}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES=""
  for ((index = 0; index < NPROC_PER_NODE; index++)); do
    [[ -z "${CUDA_VISIBLE_DEVICES}" ]] || CUDA_VISIBLE_DEVICES+=","
    CUDA_VISIBLE_DEVICES+="${index}"
  done
fi
[[ "$(visible_gpu_count)" == "${NPROC_PER_NODE}" ]] \
  || fail "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} 与 NPROC_PER_NODE=${NPROC_PER_NODE} 不一致"

(( GROUP_SIZE % NPROC_PER_NODE == 0 )) \
  || fail "GROUP_SIZE=${GROUP_SIZE} 必须整除 GPU 数 ${NPROC_PER_NODE}"
(( GROUP_SIZE / NPROC_PER_NODE >= 2 )) \
  || fail "每卡至少需要 2 个 rollout；当前 ${GROUP_SIZE}/${NPROC_PER_NODE}"
(( EVAL_SEEDS_PER_CONDITION % EVAL_ROLLOUT_BATCH_SIZE == 0 )) \
  || fail "EVAL_ROLLOUT_BATCH_SIZE 必须整除 EVAL_SEEDS_PER_CONDITION"

[[ -f "${IDS_FILE}" ]] || fail "找不到固定样本清单：${IDS_FILE}"
mapfile -t CONDITION_IDS < <(sed -e 's/[[:space:]]*$//' "${IDS_FILE}" | awk 'NF && $1 !~ /^#/ { print $1 }')
[[ "${#CONDITION_IDS[@]}" -eq 16 ]] \
  || fail "固定样本清单必须恰好包含 16 个 ID，当前 ${#CONDITION_IDS[@]}"
[[ "$(printf '%s\n' "${CONDITION_IDS[@]}" | sort -u | wc -l)" -eq 16 ]] \
  || fail "固定样本清单包含重复 ID"

[[ -d "${PREP_ROOT}" ]] || fail "找不到 overfit prep：${PREP_ROOT}"
[[ -d "${GT_ROOT}" ]] || fail "找不到 overfit GT：${GT_ROOT}"
for condition_id in "${CONDITION_IDS[@]}"; do
  [[ -f "${PREP_ROOT}/${condition_id}/actions.npy" ]] \
    || fail "缺少 prep：${condition_id}/actions.npy"
  for camera in head hand_left hand_right; do
    [[ -f "${GT_ROOT}/${condition_id}/${camera}_29_frames.mp4" ]] \
      || fail "缺少 GT：${condition_id}/${camera}_29_frames.mp4"
  done
done

mapfile -t ACTUAL_IDS < <(find "${PREP_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
[[ "${#ACTUAL_IDS[@]}" -eq 16 ]] \
  || fail "${PREP_ROOT} 必须只包含固定 16 个 condition，当前 ${#ACTUAL_IDS[@]}"
[[ "$(printf '%s\n' "${ACTUAL_IDS[@]}" | sha256sum | awk '{print $1}')" == \
   "$(printf '%s\n' "${CONDITION_IDS[@]}" | sort | sha256sum | awk '{print $1}')" ]] \
  || fail "prep 目录中的 condition 与固定清单不一致"

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)_${NPROC_PER_NODE}gpu_$$"
  OUTPUT_DIR="${DATA_ROOT}/awm_coca_overfit16_outputs/runs/${RUN_STAMP}"
fi

TRAIN_ARGS=(
  --group-size "${GROUP_SIZE}"
  --max-optimizer-steps "${MAX_OPTIMIZER_STEPS}"
  --eval-prep-root "${PREP_ROOT}"
  --eval-max-conditions 16
  --eval-every-group-steps "${EVAL_EVERY_GROUP_STEPS}"
  --eval-seeds-per-condition "${EVAL_SEEDS_PER_CONDITION}"
  --eval-rollout-batch-size "${EVAL_ROLLOUT_BATCH_SIZE}"
  --eval-seed "${EVAL_SEED}"
  "$@"
)

echo "[INFO] AWM-CoCA 16-condition overfit"
echo "[INFO] commit=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || true)"
echo "[INFO] GPUs=${NPROC_PER_NODE} (${CUDA_VISIBLE_DEVICES}), global_group=${GROUP_SIZE}, per_rank=$((GROUP_SIZE / NPROC_PER_NODE))"
echo "[INFO] prep=${PREP_ROOT}"
echo "[INFO] eval_prep=${PREP_ROOT}（与训练集完全相同）"
echo "[INFO] gt=${GT_ROOT}"
echo "[INFO] output=${OUTPUT_DIR}"
echo "[INFO] eval=group 0 baseline + every ${EVAL_EVERY_GROUP_STEPS} groups; ${EVAL_SEEDS_PER_CONDITION} seeds/condition; batch=${EVAL_ROLLOUT_BATCH_SIZE}"

COMMAND=(bash "${SCRIPT_DIR}/train_remote.sh" "${TRAIN_ARGS[@]}")
if [[ "${OVERFIT_DRY_RUN:-0}" == "1" ]]; then
  printf '[DRY-RUN] '
  printf '%q ' env \
    "DATA_DIR=${DATA_ROOT}" "PREP_DIR=${PREP_ROOT}" "GT_DIR=${GT_ROOT}" \
    "MODEL_DIR=${MODEL_DIR}" "OUTPUT_DIR=${OUTPUT_DIR}" \
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" "NPROC_PER_NODE=${NPROC_PER_NODE}" \
    "ROLLOUT_RETENTION=${ROLLOUT_RETENTION}" "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec env \
  "DATA_DIR=${DATA_ROOT}" \
  "PREP_DIR=${PREP_ROOT}" \
  "GT_DIR=${GT_ROOT}" \
  "MODEL_DIR=${MODEL_DIR}" \
  "OUTPUT_DIR=${OUTPUT_DIR}" \
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
  "NPROC_PER_NODE=${NPROC_PER_NODE}" \
  "ROLLOUT_RETENTION=${ROLLOUT_RETENTION}" \
  "${COMMAND[@]}"
