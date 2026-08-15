#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IDS_FILE="${OVERFIT16_IDS_FILE:-${REPO_ROOT}/configs/awm_coca_overfit16_ids.txt}"
PREP_SOURCE="${PREP_SOURCE:-${REPO_ROOT}/../Genie-Envisioner-V1/output/prep}"
GT_SOURCE="${GT_SOURCE:-${REPO_ROOT}/../../haoran/data/agibotworld_beta/selected_samples/samples}"
ARCHIVE="${1:-${REPO_ROOT}/../../awm_coca_overfit16.tar.gz}"

fail() {
  echo "[ERROR] $*" >&2
  exit 2
}

[[ -f "${IDS_FILE}" ]] || fail "找不到样本清单：${IDS_FILE}"
[[ -d "${PREP_SOURCE}" ]] || fail "找不到 prep 源目录：${PREP_SOURCE}"
[[ -d "${GT_SOURCE}" ]] || fail "找不到 GT 源目录：${GT_SOURCE}"
[[ ! -e "${ARCHIVE}" ]] || fail "目标压缩包已存在，不覆盖：${ARCHIVE}"

mapfile -t CONDITION_IDS < <(sed -e 's/[[:space:]]*$//' "${IDS_FILE}" | awk 'NF && $1 !~ /^#/ { print $1 }')
[[ "${#CONDITION_IDS[@]}" -eq 16 ]] || fail "样本清单必须包含 16 个 ID"

STAGING_DIR="$(mktemp -d)"
cleanup() {
  [[ -n "${STAGING_DIR:-}" && -d "${STAGING_DIR}" ]] && rm -rf -- "${STAGING_DIR}"
}
trap cleanup EXIT

PACKAGE_ROOT="${STAGING_DIR}/awm_coca_overfit16"
mkdir -p "${PACKAGE_ROOT}/prep" "${PACKAGE_ROOT}/selected_samples/samples"
cp "${IDS_FILE}" "${PACKAGE_ROOT}/condition_ids.txt"

for condition_id in "${CONDITION_IDS[@]}"; do
  [[ -f "${PREP_SOURCE}/${condition_id}/actions.npy" ]] \
    || fail "缺少 prep：${condition_id}/actions.npy"
  for camera in head hand_left hand_right; do
    [[ -f "${GT_SOURCE}/${condition_id}/${camera}_29_frames.mp4" ]] \
      || fail "缺少 GT：${condition_id}/${camera}_29_frames.mp4"
  done
  ln -s "${PREP_SOURCE}/${condition_id}" "${PACKAGE_ROOT}/prep/${condition_id}"
  ln -s "${GT_SOURCE}/${condition_id}" "${PACKAGE_ROOT}/selected_samples/samples/${condition_id}"
done

mkdir -p "$(dirname "${ARCHIVE}")"
tar --dereference -czf "${ARCHIVE}" -C "${STAGING_DIR}" awm_coca_overfit16
(
  cd "$(dirname "${ARCHIVE}")"
  sha256sum "$(basename "${ARCHIVE}")" >"$(basename "${ARCHIVE}").sha256"
)

echo "[DONE] ${ARCHIVE}"
echo "[DONE] ${ARCHIVE}.sha256"
du -h "${ARCHIVE}"
