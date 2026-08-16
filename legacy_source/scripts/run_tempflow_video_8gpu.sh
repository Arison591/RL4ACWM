#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TEMPFLOW_DATA_ROOT:?set TEMPFLOW_DATA_ROOT to the extracted overfit16 root}"
: "${TEMPFLOW_ASSET_ROOT:?set TEMPFLOW_ASSET_ROOT to the GE-Sim asset repository root}"
: "${TEMPFLOW_OUTPUT_ROOT:?set TEMPFLOW_OUTPUT_ROOT to a writable output root}"

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
if [[ "${GPU_COUNT}" -ne 8 ]]; then
  echo "Refusing launch: single_node_8gpu.yaml requires exactly 8 visible GPUs; found ${GPU_COUNT}." >&2
  exit 2
fi
echo "Refusing launch: global reward gather/DDP update is a documented migration blocker." >&2
echo "After the 2-GPU gate is implemented and passed, the intended command is:" >&2
echo "torchrun --standalone --nnodes=1 --nproc-per-node=8 -m experiments.tempflow_video.run --config configs/tempflow_video/single_node_8gpu.yaml" >&2
echo "See docs/tempflow_video/05_eight_gpu_migration.md; do not claim this configuration is validated." >&2
exit 2
