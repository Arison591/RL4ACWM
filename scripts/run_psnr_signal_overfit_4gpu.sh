#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${TEMPFLOW_OVERFIT_CONDITION_ID:-}" ]]; then
  echo "Missing TEMPFLOW_OVERFIT_CONDITION_ID (choose one ID from the overfit16 id list)" >&2
  exit 2
fi

export TEMPFLOW_CONFIG="${repo_root}/configs/psnr_signal_overfit_4gpu.yaml"
exec "${repo_root}/scripts/run_psnr_only_4gpu.sh" "$@"
