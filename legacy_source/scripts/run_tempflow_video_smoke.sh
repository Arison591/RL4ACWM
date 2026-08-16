#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TEMPFLOW_DATA_ROOT:?set TEMPFLOW_DATA_ROOT to the extracted overfit16 root}"
: "${TEMPFLOW_ASSET_ROOT:?set TEMPFLOW_ASSET_ROOT to the GE-Sim asset repository root}"
: "${TEMPFLOW_OUTPUT_ROOT:?set TEMPFLOW_OUTPUT_ROOT to a writable output root}"
exec "${REPO_ROOT}/scripts/run_tempflow_video.sh" configs/tempflow_video/smoke_single_gpu.yaml "$@"
