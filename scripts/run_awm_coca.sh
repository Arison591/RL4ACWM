#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-train}"
shift || true

cd "${REPO_ROOT}"

case "${MODE}" in
  smoke)
    exec "${PYTHON_BIN}" -m experiments.awm_coca.run_train \
      --config configs/awm_coca_train.yaml --smoke-test "$@"
    ;;
  preflight)
    exec "${PYTHON_BIN}" -m experiments.awm_coca.run_train \
      --config configs/awm_coca_train.yaml --preflight-only "$@"
    ;;
  train)
    exec "${PYTHON_BIN}" -m experiments.awm_coca.run_train \
      --config configs/awm_coca_train.yaml "$@"
    ;;
  train4)
    exec "${PYTHON_BIN}" -m torch.distributed.run --standalone \
      --nproc_per_node="${NPROC_PER_NODE:-4}" \
      -m experiments.awm_coca.run_train \
      --config configs/awm_coca_train.yaml "$@"
    ;;
  *)
    echo "Usage: $0 {smoke|preflight|train|train4} [run_train options...]" >&2
    exit 2
    ;;
esac
