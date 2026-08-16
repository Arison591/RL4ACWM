# PSNR reward fusion report

Offline input contained 60 complete groups and 360 branches. No video was generated. The prompt's
estimate of 336 branches was not used to discard the additional valid groups.

## Calibration

Raw future-only aggregate PSNR group std ranged from 0.001064 to 0.515101 dB; p01/p05/median were
0.001682/0.002347/0.046112 dB. Action std ranged from 0.001724 to 0.145422. Saved condition metadata
contains no Fast/Slow label, so that requested split is explicitly unavailable rather than inferred.
Stored per-frame PSNR recomputation was deterministic (0 dB discrepancy). The selected PSNR threshold
is 0.0002 dB (approximately one tenth of p01, rounded upward); Action uses 0.0001. No existing group
falls below either threshold, but the gate is tested with synthetic low-variance groups.

## Old versus new

Under legacy joint reward and group normalization, the RMS Action contribution was 0.96876 and the
RMS sigmoid-PSNR contribution 0.50070, a PSNR/Action ratio of 0.51685. Across all branches,
`corr(total, action)=0.35755` and `corr(total, raw PSNR)=0.81139`. These data do not reproduce the
claimed tenfold suppression; they still show that nominal 0.5/0.5 did not mean equal normalized
contribution.

With component normalization and 0.5/0.5 weights, weighted advantage RMS was 0.37062 for Action and
0.37460 for PSNR. Effective PSNR groups were 60/60 and skipped groups 0/60. Pairwise Action/PSNR
ranking conflict was 50.22%. Positive affine z-scoring and clipping do not change PSNR ordering.
The observed minimum is more than five times the threshold, so the existing set is not being amplified
from near-numerical noise.

This establishes that raw Action variance no longer suppresses PSNR advantage. It does not establish
equal gradient contribution: weighted gradient norms depend on policy Jacobians and must be measured
during a real AWM smoke/formal run.

