# Action RL Reward Audit

## Scope and baseline

This audit applies to the Action-only TempFlow reward copied into this
independent checkout from the running four-GPU baseline on 2026-08-19. The
previous merged-two-arm-centroid defect is not present: each YOLO arm class is
compared only with the command trajectory for that arm.

The relevant source hashes are recorded in `../baseline_snapshots/`; the
running baseline effective-config SHA256 is
`950e2fabbb0e4fa725833ec9525bf464d9ca27e4dd70812acdf7cd50f8b1e386`.

## Command calculation chain

1. `actions.npy` contains left pose at columns `0:7` and right pose at
   `8:15`. `commanded_trajectory()` converts the quaternion to SE(3), applies
   `inv(extrinsic_head) @ pose @ z_offset(0.23m)`, and projects the EEF origin
   through `intrinsic_head` at the generated video resolution.
2. YOLO-World class `0` is the left EEF and class `1` is the right EEF. For
   every generated video frame, the highest-confidence box of the requested
   class contributes its bbox centre. The detector tracker is reset before and
   after every video; tracking persistence is disabled because score selection
   does not use identities.
3. For each arm and every jointly valid frame except frame zero,
   `e_arm,t = ||YOLO_arm,t - command_arm,t||_2` in pixels.
4. `left_arm_raw_error` and `right_arm_raw_error` are their per-arm means.
   `per_frame_raw_error` records the mean across available arms at every frame.
   `combined_raw_command_error` is the mean over all valid arm-frame errors.
5. The legacy evaluator also computes
   `af_fdce_ate_norm = combined_raw_command_error / sqrt(H^2 + W^2)` and
   `final_command_component = 1 - clip(af_fdce_ate_norm / 0.2, 0, 1)`.

For the current 640x480 reward video, the image diagonal is 800 px. The
legacy mapping is therefore `1 - raw_error_px / 160` until 160 px, followed by
zero clipping. The observed command component range `0.699..0.715` maps to raw
error about `48.2..45.6 px`: a roughly 2.6 px within-group difference. Thus
the narrow range is not caused by saturation in this range, but raw command
error itself needs to be the training ranking signal and must be gated against
the measured evaluator noise floor.

## Training versus evaluation value

The compatibility evaluation value remains `final_command_component`.
The corrected training component will be `-combined_raw_command_error` after
the repeatability audit establishes a fixed noise floor. It preserves arm
identity, target command, temporal alignment, and ordering, while avoiding an
unnecessary clipped presentation transform in the GRPO reward path.

## Semantic checks

`awm_source/tests/test_action_command_reward_semantics.py` covers perfect
two-arm matching, small and large translation monotonicity, arm swapping,
time reversal, single-arm failure metadata, and A-B-A stateless reuse. The
existing `test_reward_runner_action_metrics.py` confirms that reward_runner
passes each arm's matching command rather than a shared centroid.
