import torch

from tempflow_video.runtime.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_resume(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    path = save_checkpoint(tmp_path / "checkpoint_1", policy=model, optimizer=optimizer,
                           trainer_state={"optimizer_step": 1, "policy_version": 1})
    with torch.no_grad():
        model.weight.add_(10)
    state = load_checkpoint(path, policy=model, optimizer=optimizer)
    assert state == {"optimizer_step": 1, "policy_version": 1}
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value)

