#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TEMPFLOW_CONFIG="${TEMPFLOW_CONFIG:-${repo_root}/configs/da3_mono_signal_overfit_4gpu.yaml}"
exec "${repo_root}/scripts/run_da3_mono_signal_smoke_4gpu.sh" "$@"
