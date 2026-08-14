#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${1:-${OUTPUT_DIR:-${REPO_ROOT}/outputs/awm_coca_remote}}"

if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
fi

CHECKPOINT_ROOT="${OUTPUT_DIR}/checkpoints"
[[ -d "${CHECKPOINT_ROOT}" ]] || {
  echo "[ERROR] checkpoint 目录不存在: ${CHECKPOINT_ROOT}" >&2
  exit 2
}

LATEST_CHECKPOINT="$(find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint_*' -print | sort -V | tail -n 1)"
[[ -n "${LATEST_CHECKPOINT}" && -f "${LATEST_CHECKPOINT}/COMPLETE" ]] || {
  echo "[ERROR] 找不到带 COMPLETE 标记的完整 checkpoint" >&2
  exit 2
}

CHECKPOINT_NAME="$(basename "${LATEST_CHECKPOINT}")"
STEP="${CHECKPOINT_NAME#checkpoint_}"
TRANSFER_DIR="${OUTPUT_DIR}/transfer"
mkdir -p "${TRANSFER_DIR}"
ARCHIVE="${TRANSFER_DIR}/awm_coca_${CHECKPOINT_NAME}_$(date +%Y%m%d_%H%M%S).tar.gz"

items=("checkpoints/${CHECKPOINT_NAME}")
for optional in metrics logs dataset; do
  [[ -e "${OUTPUT_DIR}/${optional}" ]] && items+=("${optional}")
done

tar -C "${OUTPUT_DIR}" -czf "${ARCHIVE}" "${items[@]}"
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"

echo "[DONE] 已打包 step=${STEP} 的最新完整 checkpoint、训练日志和指标"
echo "[DONE] ${ARCHIVE}"
echo "[DONE] ${ARCHIVE}.sha256"
