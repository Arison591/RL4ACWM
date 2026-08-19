#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${TEMPFLOW_PYTHON:-python}"
CONFIG="${1:-configs/tempflow_video/smoke_single_gpu.yaml}"
shift || true

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
exec "${PYTHON_BIN}" -m experiments.tempflow_video.run --config "${CONFIG}" "$@"
