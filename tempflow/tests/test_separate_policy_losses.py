import torch

from tempflow_video.core.policy_objective import component_policy_objective


def test_separate_policy_losses():
    logp = torch.tensor([0.1, -0.1], requires_grad=True)
    kwargs = dict(log_probs=logp, old_log_probs=torch.zeros(2), noise_weights=torch.ones(2),
                  policy_means=logp.reshape(2, 1), reference_means=torch.zeros(2, 1),
                  transition_stds=torch.ones(2), clip_range=0.2, kl_beta=0)
    out = component_policy_objective(action_advantages=torch.tensor([-1., 1.]),
                                     psnr_advantages=torch.tensor([1., -1.]), **kwargs)
    assert out.action_policy_loss.item() != out.psnr_policy_loss.item()
    torch.testing.assert_close(out.total_loss, out.weighted_action_policy_loss + out.weighted_psnr_policy_loss)

