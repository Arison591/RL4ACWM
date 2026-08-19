import pytest
import torch

from experiments.tempflow_video.policy import ReferencePolicyAdapter
from tests.tempflow_test_utils import ToyVideoPolicy


def test_reference_adapter_detects_any_base_parameter_mutation():
    policy = ToyVideoPolicy()
    reference = ReferencePolicyAdapter(policy)
    reference.assert_unchanged()

    with torch.no_grad():
        policy.policy_model.base.add_(1.0)

    with pytest.raises(RuntimeError, match="reference policy parameter changed"):
        reference.assert_unchanged()


def test_policy_parameter_is_not_part_of_reference_snapshot():
    policy = ToyVideoPolicy()
    reference = ReferencePolicyAdapter(policy)
    with torch.no_grad():
        policy.policy_model.delta.add_(1.0)
    reference.assert_unchanged()
