# TempFlow + DA3 Mono reward

This branch adds Haoran's DA3 Mono pseudo-depth reward as an optional TempFlow
terminal reward. It does not change the rollout policy, timestep selection,
PPO objective, optimizer, or reference KL.

## Data parity with PSNR

PSNR and DA3 Mono consume exactly the same rollout assets:

- cameras: `head`, `hand_left`, `hand_right`;
- generated files: `<branch>/head_color.mp4`, `hand_left_color.mp4`,
  `hand_right_color.mp4`;
- GT files: the existing `reward.gt_video_templates`;
- scored interval: future frames 4 through 28 inclusive (25 frames).

The only change is the representation used for scoring. PSNR compares RGB
pixels. Mono independently sends every GT/generated frame through frozen
DA3-BASE and compares the resulting pseudo-depth maps. It is not Joint0 or
Joint1 multi-view inference.

For each view, generated inverse depth is affine-aligned to GT inverse depth
with one scale/shift shared by the whole 25-frame clip. The default robust loss
uses confidence-weighted Huber error, trims the largest 5% errors, excludes a
2% border, and mixes full-frame and GT-motion-region errors:

```text
E_view = 0.65 * E_full + 0.35 * E_dynamic
E_mono = mean(E_head, E_hand_left, E_hand_right)
R_mono = -E_mono
```

GT pseudo-depth and its dynamic mask are cached in memory per process and
condition. Generated pseudo-depth is always recomputed for each branch.

## Offline model requirement

The training job never downloads model code or weights. Supply explicit local
paths:

```bash
export DA3_SOURCE_ROOT=/path/to/Depth-Anything-3/src
export DA3_MODEL_PATH=/path/to/models--depth-anything--DA3-BASE/snapshots/<revision>
```

The launcher sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Preflight
checks both paths; `--load-model-preflight` additionally loads AWM and DA3 on
the same GPU to catch dependency and memory failures before rollout collection.

## Four-GPU signal smoke

Choose the same single condition used by the PSNR signal smoke and export the
standard AWM/TempFlow data paths, then run:

```bash
export TEMPFLOW_OVERFIT_CONDITION_ID=<condition-id>
scripts/run_da3_mono_signal_smoke_4gpu.sh
```

The committed configuration is
`configs/da3_mono_signal_smoke_4gpu.yaml`. It keeps timestep 2, 12 global
branches, learning rate `1e-5`, KL beta `0.01`, and two inner PPO passes. It is
limited to 10 optimizer steps. The provisional `min_group_std=1e-5` is only a
degeneracy gate and must be calibrated from real 12-branch Mono groups before
a longer run.

Each `reward.json` records the total negative depth error, per-view full and
dynamic errors, alignment values, confidence/valid ratios, per-frame errors,
model provenance, and the exact score configuration.

An end-to-end local check on two existing TempFlow branches for
`001_task_327_episode_684757` produced rewards `-0.0438563` and `-0.0456469`
(population std `8.95e-4`). This validates decoding, offline DA3 inference,
GT-cache reuse and reward direction, but two branches are not enough to claim
that the committed 12-branch variance threshold is calibrated.

## Provenance

The low-level `legacy_source/da3_reward` package was copied from the local
Haoran working tree on 2026-08-20. The source directory was untracked at its
repository HEAD, so the vendoring note records that fact instead of assigning
the files a false Git revision. DA3 source and model weights remain external.
