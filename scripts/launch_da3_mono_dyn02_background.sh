#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TEMPFLOW_CONFIG="${repo_root}/configs/da3_mono_dyn02_signal_overfit_4gpu.yaml"
export WANDB_PROJECT="${WANDB_PROJECT:-awm-coca}"
export WANDB_NAME="${WANDB_NAME:-da3-mono-dyn02-full08-4gpu}"
exec "${repo_root}/scripts/launch_da3_mono_background.sh" "$@"
