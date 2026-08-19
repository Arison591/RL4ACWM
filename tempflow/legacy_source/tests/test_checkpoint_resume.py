import torch
from peft import LoraConfig, get_peft_model

from experiments.tempflow_video.checkpointing import (
    _policy_state,
    load_tempflow_checkpoint,
    restore_tempflow_checkpoint,
    save_tempflow_checkpoint,
)


def test_peft_checkpoint_contains_only_adapter_state():
    model = get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(4, 4)),
        LoraConfig(r=2, lora_alpha=4, target_modules=["0"]),
    )
    kind, state = _policy_state(model)
    assert kind == "peft"
    assert state
    assert all("lora_" in name for name in state)
    assert sum(value.numel() for value in state.values()) < sum(
        value.numel() for value in model.state_dict().values()
    )


def test_tempflow_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    checkpoint = save_tempflow_checkpoint(
        tmp_path,
        step=1,
        policy=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        trainer_state={"optimizer_step": 1, "policy_version": 1},
        config={"experiment": {"mode": "tempflow_full"}},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)

    payload = load_tempflow_checkpoint(checkpoint)
    trainer_state = restore_tempflow_checkpoint(
        payload, policy=model, optimizer=optimizer, lr_scheduler=scheduler
    )

    assert trainer_state == {"optimizer_step": 1, "policy_version": 1}
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])
    assert payload["config"]["experiment"]["mode"] == "tempflow_full"


def test_incomplete_checkpoint_is_rejected(tmp_path):
    broken = tmp_path / "checkpoint_2"
    broken.mkdir()
    try:
        load_tempflow_checkpoint(broken)
    except ValueError as exc:
        assert "incomplete checkpoint" in str(exc)
    else:
        raise AssertionError("incomplete checkpoint was accepted")
