# Single-GPU smoke report

Fourteen CPU/unit tests pass, including exact transition parity, upstream cleanliness/import, legacy
reward forwarding, PSNR scope/aggregation/raw-dB use, independent normalization, low-variance skip,
separate losses, nonzero PSNR gradient, reference freeze, optimizer mutation and checkpoint/resume.

A one-update algebraic component-loss smoke also ran on the available CUDA device. It produced finite,
nonzero Action and PSNR term gradients, changed policy parameters, preserved the reference and restored
the checkpoint. Unweighted gradient norms were 0.707102 (Action) and 0.707036 (PSNR); weighted norms
were 0.353551 and 0.353518, with cosine 1.0 in this deliberately tiny construction. This smoke
deliberately did not generate video or call the expensive Action stack.

The requested full AWM path (generation → real Action → PSNR → component losses → KL → backward) is
not claimed complete: the supplied official audit commit is unavailable from configured origin, and the
standalone sampler/condition integration has not yet passed generation parity. Starting a costly model
load before resolving those gates would violate parity-first ordering.
