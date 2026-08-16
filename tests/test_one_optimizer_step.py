import torch

from tempflow_video.core.policy_objective import component_policy_objective
from tempflow_video.core.trainer import optimizer_step


def test_one_optimizer_step_has_psnr_gradient():
    model = torch.nn.Linear(1, 2, bias=False)
    x = torch.ones(1, 1)
    logits = model(x).flatten()
    loss = component_policy_objective(log_probs=logits, old_log_probs=torch.zeros(2),
        action_advantages=torch.tensor([-1., 1.]), psnr_advantages=torch.tensor([1., -0.5]),
        noise_weights=torch.ones(2), policy_means=logits.reshape(2, 1),
        reference_means=logits.detach().reshape(2, 1), transition_stds=torch.ones(2),
        clip_range=0.2, kl_beta=0.01)
    before = model.weight.detach().clone()
    metrics = optimizer_step(loss, parameters=list(model.parameters()),
                             optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3))
    assert metrics["psnr_policy_grad_norm"] > 0
    assert not torch.equal(before, model.weight)
