#!/usr/bin/env bash
set -euo pipefail
if [[ "${APPROVE_FORMAL_OVERFIT:-0}" != 1 ]]; then
  echo "Formal overfit is intentionally gated. Review configs/overfit16_component_advantage.yaml and set APPROVE_FORMAL_OVERFIT=1 only after approval."
  exit 2
fi
echo "Runner integration is not yet cleared; no training was started."
exit 3

