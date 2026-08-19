# PSNR reward design

PSNR is a visual-fidelity reward, not a geometry or action-correctness metric. Action reward is kept
numerically unchanged behind the legacy adapter.

The module emits per-frame/per-view PSNR, per-view full and future-only values, aggregate full and
future-only dB, and the exact legacy sigmoid. Aggregation preserves the audited order: average frames
inside each view, then `0.6 * mean-view + 0.4 * worst-view`. Training consumes raw future-only aggregate
dB. `history_frames` is mandatory runtime metadata and is not hard-coded. Fixed evaluation reports both
scopes.

Legacy audit correction: the old implementation was configured for frames 4..28, not all 29 frames.
`legacy_psnr_sigmoid` preserves that configured range independently of the new full-clip metric.

Within each branch group, Action and PSNR are population-standardized independently, then independently
clipped to [-1, 1]. A component below its calibrated minimum std becomes all-zero. If both components
are skipped, the group must not update. No reward noise is added and the combined advantage is never
standardized again.

The policy objective computes independent Action and PSNR PPO/GRPO losses, weights them 0.5/0.5, and
adds weighted reference KL. Component normalization aligns numerical advantage scale; it does not imply
equal parameter gradients. Debug logging therefore uses side-effect-free `autograd.grad` calls to report
weighted term norms and Action/PSNR gradient cosine.
