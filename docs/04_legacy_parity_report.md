# Legacy parity report

Transition parity is tested by loading the original uncommitted `dynamics.py` directly and comparing
mean, variance/std, sampled next latent, old/new-compatible log-probability and RF noise scale at zero
tolerance. Reward adapter parity verifies identity return and exact argument forwarding.

Existing real reward artifacts report direct-vs-adapter parity at `1e-8` with zero mismatches. The
standalone PSNR implementation additionally receives numeric parity tests before becoming a training
input. Full AWM generation parity requires external model/data assets and is reported separately from
CPU unit tests.
