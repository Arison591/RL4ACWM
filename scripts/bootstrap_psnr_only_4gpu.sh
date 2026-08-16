#!/usr/bin/env bash
set -euo pipefail

install_root="${1:-${PWD}/tempflow_psnr_workspace}"
repo_url="${TEMPFLOW_GIT_URL:-https://github.com/Arison591/RL4ACWM.git}"
branch="${TEMPFLOW_GIT_BRANCH:-agent/tempflow-psnr-only-4gpu}"
upstream_commit="dce69e48a952449e873a791812e506df878bc8a9"
standalone_root="${install_root}/RL4ACWM-tempflow-video"
upstream_root="${install_root}/RL4ACWM-upstream-clean"

mkdir -p "${install_root}"
if [[ -e "${standalone_root}" ]]; then
  echo "Refusing to overwrite existing path: ${standalone_root}" >&2
  exit 2
fi
if [[ -e "${upstream_root}" ]]; then
  echo "Refusing to overwrite existing path: ${upstream_root}" >&2
  exit 2
fi

git clone --single-branch --branch "${branch}" "${repo_url}" "${standalone_root}"
git clone --no-checkout "${repo_url}" "${upstream_root}"
git -C "${upstream_root}" checkout --detach "${upstream_commit}"
if [[ -n "$(git -C "${upstream_root}" status --porcelain=v1 -uall)" ]]; then
  echo "Fresh upstream checkout is unexpectedly dirty" >&2
  exit 2
fi

mkdir -p "${install_root}/outputs"
cat <<EOF
Created isolated workspace:
  TempFlow: ${standalone_root}
  clean AWM: ${upstream_root}

Set the two machine-specific asset/data paths, then run:
  export AWM_UPSTREAM_ROOT='${upstream_root}'
  export AWM_ASSET_ROOT='/path/to/Genie-Envisioner-V1'
  export TEMPFLOW_DATA_ROOT='/path/to/awm_coca_overfit16'
  export TEMPFLOW_OUTPUT_ROOT='${install_root}/outputs'
  cd '${standalone_root}'
  scripts/run_psnr_only_4gpu.sh
EOF
