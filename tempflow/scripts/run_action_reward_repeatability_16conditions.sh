#!/usr/bin/env bash
set -euo pipefail

saved_run="${1:?usage: run_action_reward_repeatability_16conditions.sh <run-dir> [output-dir]}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${2:-${TEMPFLOW_OUTPUT_ROOT:?set TEMPFLOW_OUTPUT_ROOT}/action_component_correction_4gpu/repeatability/$(date -u +%Y%m%dT%H%M%SZ)}"
config="${saved_run}/effective_config.yaml"

if [[ ! -f "${config}" ]]; then
  echo "Missing effective config: ${config}" >&2
  exit 2
fi
for variable in AWM_UPSTREAM_ROOT AWM_ASSET_ROOT; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing required environment variable: ${variable}" >&2
    exit 2
  fi
done

export AWM_MODEL_ROOT="${AWM_MODEL_ROOT:-${AWM_ASSET_ROOT}/awm_coca_models}"
export PYTHONPATH="${repo_root}/src:${repo_root}/legacy_source:${AWM_UPSTREAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${output_dir}"
IFS=',' read -r -a devices <<< "${REPEATABILITY_GPUS:-4,5,6,7}"
if [[ "${#devices[@]}" != 4 ]]; then
  echo "REPEATABILITY_GPUS must list exactly four GPU IDs" >&2
  exit 2
fi
for worker in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${devices[worker]}" \
    "${PYTHON_BIN:-python}" "${repo_root}/scripts/action_reward_repeatability.py" \
    --config "${config}" --saved-run "${saved_run}" --output-dir "${output_dir}" \
    --repeats 3 --limit 16 --distinct-conditions --offset "${worker}" --stride 4 --worker-id "${worker}" \
    >"${output_dir}/worker${worker}.log" 2>&1 &
done
wait
"${PYTHON_BIN:-python}" "${repo_root}/scripts/merge_reward_noise_floor.py" \
  --input-dir "${output_dir}" --output "${output_dir}/reward_noise_floor.json"
echo "${output_dir}"
