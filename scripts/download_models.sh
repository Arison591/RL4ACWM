#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/checkpoints}"

if [[ "${MODEL_DIR}" != /* ]]; then
  MODEL_DIR="${REPO_ROOT}/${MODEL_DIR}"
fi

COSMOS_REPO="nvidia/Cosmos-Predict2-2B-Video2World"
COSMOS_REV="f50c09f5d8ab133a90cac3f4886a6471e9ba3f18"
SAM3_MODEL_REPO="facebook/sam3"
SAM3_MODEL_REV="3c879f39826c281e95690f02c7821c4de09afae7"
YOLO_REPO="agibot-world/EWMBench-model"
YOLO_REV="85a49f8118ec656c9d511c5236582ac7fde16fbc"

GESIM_REPO_REV="1422f1783e5eed8e00925d3ce9ba3a0ba59e84df"
GESIM_SHA256="0e49bbe4e83c2b6e380e0e2215f8f257ac760498b772b20e52f37a40b6649f8d"
COW_MODEL_REV="f4633e5671c5f19ea8500943869f4c975b605fde"
COW_MODEL_SHA256="5b46c6dac1d9f9f5944371176f5aedec13522db4ea86d998f3445b0fd37ab784"
SAM3_SHA256="9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
YOLO_SHA256="6f464feacf6d64bbd1d9318b05444d4c9ff301b335291cae2d808cfca5e6b257"

SAM3_SOURCE_REV="6dbb02bd38288df755dfa1378000a861e65b84f6"
COW_SOURCE_REV="1454f20045d3b514e5b8417907152677f3dba621"

for command_name in git curl sha256sum; do
  command -v "${command_name}" >/dev/null || {
    echo "[ERROR] 缺少命令: ${command_name}" >&2
    exit 2
  }
done

HF_BIN="${HF_BIN:-$(command -v hf || true)}"
if [[ -z "${HF_BIN}" ]]; then
  echo "[ERROR] 找不到 hf CLI；请先 conda activate genie-psnr" >&2
  exit 2
fi

mkdir -p "${MODEL_DIR}" "${REPO_ROOT}/third_party"

clone_pinned() {
  local name="$1" url="$2" revision="$3" destination="$4"
  if [[ -d "${destination}/.git" ]]; then
    local actual
    actual="$(git -C "${destination}" rev-parse HEAD)"
    if [[ "${actual}" != "${revision}" ]]; then
      if [[ -n "$(git -C "${destination}" status --porcelain)" ]]; then
        echo "[ERROR] ${name} 源码有本地修改，无法切换 revision: ${destination}" >&2
        exit 2
      fi
      git -C "${destination}" fetch --depth 1 origin "${revision}"
      git -C "${destination}" checkout --detach "${revision}"
    fi
  else
    if [[ -e "${destination}" && -n "$(find "${destination}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
      echo "[ERROR] ${destination} 已存在但不是 Git 仓库，拒绝覆盖" >&2
      exit 2
    fi
    git clone --filter=blob:none --no-checkout "${url}" "${destination}"
    git -C "${destination}" fetch --depth 1 origin "${revision}"
    git -C "${destination}" checkout --detach "${revision}"
  fi
  local actual
  actual="$(git -C "${destination}" rev-parse HEAD)"
  [[ "${actual}" == "${revision}" ]] || {
    echo "[ERROR] ${name} revision 不一致: ${actual}" >&2
    exit 2
  }
  echo "[OK] ${name} source @ ${actual}"
}

apply_patch_once() {
  local name="$1" repository="$2" patch_file="$3"
  if git -C "${repository}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
    echo "[OK] ${name} 兼容补丁已存在"
    return
  fi
  git -C "${repository}" apply --check "${patch_file}"
  git -C "${repository}" apply "${patch_file}"
  echo "[OK] ${name} 兼容补丁已应用"
}

verify_sha256() {
  local file="$1" expected="$2"
  [[ -s "${file}" ]] || return 1
  local actual
  actual="$(sha256sum "${file}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]]
}

download_checked() {
  local name="$1" url="$2" destination="$3" expected="$4"
  mkdir -p "$(dirname "${destination}")"
  if verify_sha256 "${destination}" "${expected}"; then
    echo "[SKIP] ${name} 已存在且 SHA256 正确"
    return
  fi
  if [[ -e "${destination}" ]]; then
    local backup="${destination}.invalid.$(date +%Y%m%d_%H%M%S)"
    mv "${destination}" "${backup}"
    echo "[WARN] 原文件校验失败，已保留为 ${backup}"
  fi
  local partial="${destination}.part"
  echo "[DOWNLOAD] ${name} -> ${destination}"
  curl -fL --retry 5 --retry-delay 5 --continue-at - --output "${partial}" "${url}"
  if ! verify_sha256 "${partial}" "${expected}"; then
    echo "[ERROR] ${name} SHA256 校验失败: ${partial}" >&2
    exit 2
  fi
  mv "${partial}" "${destination}"
  echo "[OK] ${name} SHA256=${expected}"
}

clone_pinned \
  "SAM3" "https://github.com/facebookresearch/sam3.git" "${SAM3_SOURCE_REV}" \
  "${REPO_ROOT}/third_party/sam3"
apply_patch_once \
  "SAM3" "${REPO_ROOT}/third_party/sam3" "${SCRIPT_DIR}/patches/sam3_runtime.patch"

clone_pinned \
  "CoWTracker" "https://github.com/facebookresearch/cowtracker.git" "${COW_SOURCE_REV}" \
  "${REPO_ROOT}/third_party/cowtracker"
git -C "${REPO_ROOT}/third_party/cowtracker" submodule update --init --recursive
apply_patch_once \
  "CoWTracker" "${REPO_ROOT}/third_party/cowtracker" "${SCRIPT_DIR}/patches/cowtracker_a100.patch"

echo "[LICENSE] CoWTracker 模型为 CC-BY-NC-4.0，仅限符合其许可证的用途。"

echo "[DOWNLOAD] ${COSMOS_REPO} @ ${COSMOS_REV}"
"${HF_BIN}" download "${COSMOS_REPO}" \
  --revision "${COSMOS_REV}" \
  --include "model_index.json" "scheduler/*" "text_encoder/*" "tokenizer/*" "vae/*" \
  --local-dir "${MODEL_DIR}/Cosmos-Predict2-2B-Video2World"

echo "[DOWNLOAD] ${SAM3_MODEL_REPO}/sam3.pt @ ${SAM3_MODEL_REV}"
if ! "${HF_BIN}" download "${SAM3_MODEL_REPO}" sam3.pt \
  --revision "${SAM3_MODEL_REV}" --local-dir "${MODEL_DIR}"; then
  echo "[ERROR] SAM3 是 gated 模型：请先在 Hugging Face 接受使用条款，并设置 HF_TOKEN 或执行 hf auth login。" >&2
  exit 2
fi
verify_sha256 "${MODEL_DIR}/sam3.pt" "${SAM3_SHA256}" || {
  echo "[ERROR] SAM3 权重 SHA256 校验失败" >&2
  exit 2
}

echo "[DOWNLOAD] ${YOLO_REPO}/yoloworld-EWMBench-v0.1.pt @ ${YOLO_REV}"
"${HF_BIN}" download "${YOLO_REPO}" yoloworld-EWMBench-v0.1.pt \
  --revision "${YOLO_REV}" --local-dir "${MODEL_DIR}"
verify_sha256 "${MODEL_DIR}/yoloworld-EWMBench-v0.1.pt" "${YOLO_SHA256}" || {
  echo "[ERROR] YOLO-World 权重 SHA256 校验失败" >&2
  exit 2
}

download_checked \
  "GE-Sim Cosmos" \
  "https://www.modelscope.cn/models/agibot_world/Genie-Envisioner/resolve/${GESIM_REPO_REV}/ge_sim_cosmos_v0.1.safetensors" \
  "${MODEL_DIR}/gesim/ge_sim_cosmos_v0.1.safetensors" \
  "${GESIM_SHA256}"

download_checked \
  "CoWTracker" \
  "https://www.modelscope.cn/models/facebook/cowtracker/resolve/${COW_MODEL_REV}/cowtracker_model.pth" \
  "${MODEL_DIR}/cowtracker/cowtracker_model.pth" \
  "${COW_MODEL_SHA256}"

required_files=(
  "${MODEL_DIR}/Cosmos-Predict2-2B-Video2World/model_index.json"
  "${MODEL_DIR}/Cosmos-Predict2-2B-Video2World/scheduler/scheduler_config.json"
  "${MODEL_DIR}/Cosmos-Predict2-2B-Video2World/text_encoder/model.safetensors.index.json"
  "${MODEL_DIR}/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.json"
  "${MODEL_DIR}/Cosmos-Predict2-2B-Video2World/vae/diffusion_pytorch_model.safetensors"
  "${MODEL_DIR}/gesim/ge_sim_cosmos_v0.1.safetensors"
  "${MODEL_DIR}/sam3.pt"
  "${MODEL_DIR}/cowtracker/cowtracker_model.pth"
  "${MODEL_DIR}/yoloworld-EWMBench-v0.1.pt"
)
for required in "${required_files[@]}"; do
  [[ -s "${required}" ]] || {
    echo "[ERROR] 模型资产不完整: ${required}" >&2
    exit 2
  }
done

echo "[DONE] 模型、reward 权重和第三方源码已准备完成"
echo "[DONE] MODEL_DIR=${MODEL_DIR}"
