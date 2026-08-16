# Legacy parity report

Transition parity is tested by loading the original uncommitted `dynamics.py` directly and comparing
mean, variance/std, sampled next latent, old/new-compatible log-probability and RF noise scale at zero
tolerance. Reward adapter parity verifies identity return and exact argument forwarding.

Existing real reward artifacts report direct-vs-adapter parity at `1e-8` with zero mismatches. The
standalone PSNR implementation additionally receives numeric parity tests before becoming a training
input. Full AWM generation parity requires external model/data assets and is reported separately from
CPU unit tests.

On 2026-08-16, real base generation was rerun once from the experimental source checkout and once from
the pinned clean upstream using the same effective configuration, first condition and seed 123456.
All 16 trajectory hashes and all three encoded-view MP4 SHA-256 values were identical.

The saved real Action/joint parity report for the same condition remains `ok=true` at `1e-8` with zero
errors. A fresh recomputation was attempted: SAM and YOLO initialization completed, but CoWTracker then
required unavailable `xformers`. No numeric mismatch was observed; the fresh run did not reach a result.
