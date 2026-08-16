#!/usr/bin/env bash
set -euo pipefail
: "${AWM_UPSTREAM_ROOT:?set AWM_UPSTREAM_ROOT to the pinned clean checkout}"
export WANDB_MODE=offline
export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src"
python tools/audit_upstream.py
python tools/smoke_component_update.py --updates "${SMOKE_UPDATES:-1}"

