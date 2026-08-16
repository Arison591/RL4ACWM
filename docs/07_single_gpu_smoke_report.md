# Single-GPU smoke report

Fourteen CPU/unit tests pass, including exact transition parity, upstream cleanliness/import, legacy
reward forwarding, PSNR scope/aggregation/raw-dB use, independent normalization, low-variance skip,
separate losses, nonzero PSNR gradient, reference freeze, optimizer mutation and checkpoint/resume.

A one-update algebraic component-loss smoke also ran on the available CUDA device. It produced finite,
nonzero Action and PSNR term gradients, changed policy parameters, preserved the reference and restored
the checkpoint. Unweighted gradient norms were 0.707102 (Action) and 0.707036 (PSNR); weighted norms
were 0.353551 and 0.353518, with cosine 1.0 in this deliberately tiny construction. This smoke
deliberately did not generate video or call the expensive Action stack.

## Real PSNR-only AWM smoke (2026-08-16)

The remaining engineering gate was run without Action reward: one real condition, branch timestep 2,
two independent branches and one optimizer update. The runner imported only the primary
`src/tempflow_video` package plus the pinned clean upstream; an import-path audit found zero
`legacy_source` modules.

- Future-only aggregate PSNR: 20.403460 and 20.630714 dB.
- Full aggregate PSNR: 22.426037 and 22.617287 dB.
- PSNR advantages: -0.999991 and +0.999991.
- Real LoRA PSNR policy gradient norm: 1.47153e-4.
- Total gradient norm before clipping: 1.47181e-4.
- Changed LoRA tensors: 224/448; frozen reference parameters were unchanged.
- Adapter checkpoint reload was separately verified across all 448 tensors at optimizer step 1.

The scalar policy loss and initial KL both print as zero at ratio=1/reference equality, which is expected:
opposite group advantages cancel in value while their per-branch score-function derivatives do not.
The nonzero gradient and changed LoRA tensors are the relevant assertions.
