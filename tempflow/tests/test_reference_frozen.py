import torch

from tempflow_video.core.reference_kl import FrozenReference


def test_reference_frozen():
    module = torch.nn.Linear(2, 1)
    reference = FrozenReference(module)
    assert not any(p.requires_grad for p in module.parameters())
    reference.assert_unchanged()
    with torch.no_grad():
        module.weight.add_(1)
    try:
        reference.assert_unchanged()
    except RuntimeError:
        pass
    else:
        raise AssertionError("mutation was not detected")

