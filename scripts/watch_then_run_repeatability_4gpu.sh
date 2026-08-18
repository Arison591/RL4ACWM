#!/usr/bin/env bash
set -euo pipefail

pid="${1:?usage: watch_then_run_repeatability_4gpu.sh <training-pid>}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
saved_run="${SAVED_ACTION_RUN:-/hpc2hdd/home/bohantan/jhupload/hr_data/tempflow_outputs/action_signal_scalar1e5_4gpu/runs/20260818T153409Z}"
effective_config="${saved_run}/effective_config.yaml"
output_root="${ACTION_REPEATABILITY_OUTPUT_ROOT:-/hpc2hdd/home/bohantan/jhupload/hr_data/tempflow_outputs/action_component_correction_4gpu/repeatability}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="${output_root}/${stamp}"

while kill -0 "${pid}" 2>/dev/null; do
  sleep 60
done
if [[ ! -f "${effective_config}" ]]; then
  echo "Missing saved-run config: ${effective_config}" >&2
  exit 2
fi

export AWM_ASSET_ROOT="${AWM_ASSET_ROOT:-/hpc2ssd/JH_DATA/spooler/bohantan/.upload/hr_data}"
export AWM_MODEL_ROOT="${AWM_MODEL_ROOT:-/hpc2ssd/JH_DATA/spooler/bohantan/.upload/hr_data/awm_coca_models}"
export PYTHONPATH="${repo_root}/src:${repo_root}/legacy_source:${repo_root}/../awm_source${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${output_dir}"
for worker in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$((worker + 4))" \
    /hpc2hdd/home/bohantan/.conda/envs/genie-psnr/bin/python \
    "${repo_root}/scripts/action_reward_repeatability.py" \
    --config "${effective_config}" --saved-run "${saved_run}" --output-dir "${output_dir}" \
    --repeats 3 --limit 48 --offset "${worker}" --stride 4 --worker-id "${worker}" \
    >"${output_dir}/worker${worker}.log" 2>&1 &
done
wait
/hpc2hdd/home/bohantan/.conda/envs/genie-psnr/bin/python \
  "${repo_root}/scripts/merge_reward_noise_floor.py" \
  --input-dir "${output_dir}" --output "${output_dir}/reward_noise_floor.json"
echo "${output_dir}"
