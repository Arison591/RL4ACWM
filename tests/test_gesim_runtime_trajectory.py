from __future__ import annotations

from types import SimpleNamespace

import torch

from experiments.awm_coca.gesim_runtime import (
    PersistentGeSimRuntime,
    PreparedGeSimCondition,
)


class _FakePipeline:
    def __init__(self) -> None:
        self.end_inputs = None

    def infer(self, **kwargs):
        self.end_inputs = kwargs["callback_on_step_end_tensor_inputs"]
        batch_views = kwargs["video"].shape[0]
        start = torch.zeros(batch_views, 2, 1, 1, 1)
        kwargs["callback_on_step_start"](
            self,
            0,
            torch.tensor(1.0),
            {
                "latents": start,
                "conditioning_latents": torch.zeros_like(start),
                "cond_indicator": torch.zeros_like(start),
                "cond_mask": torch.zeros_like(start),
                "padding_mask": torch.zeros_like(start),
                "cond_to_concat": torch.zeros_like(start),
                "prompt_embeds": torch.zeros(batch_views, 1, 1),
            },
        )
        latents = start
        for index in range(kwargs["num_inference_steps"]):
            denoised = torch.full_like(start, float(index + 10))
            latents = torch.full_like(start, float(index + 1))
            kwargs["callback_on_step_end"](
                self,
                index,
                torch.tensor(float(index)),
                {"latents": latents, "denoised": denoised},
            )
        return {
            "frames": torch.zeros(
                batch_views, 3, kwargs["num_frames"], 2, 2
            )
        }


def test_rollout_batch_captures_predicted_x0_for_every_reverse_step(tmp_path, monkeypatch):
    runtime = PersistentGeSimRuntime.__new__(PersistentGeSimRuntime)
    runtime.config = {
        "rollout": {
            "history_frames": 1,
            "future_frames": 1,
            "reverse_denoise_steps": 2,
        }
    }
    runtime.device = torch.device("cpu")
    runtime.args = SimpleNamespace(data={"train": {"valid_cam": ["head"]}})
    runtime.pipe = _FakePipeline()
    runtime.policy_version = 0
    monkeypatch.setattr("experiments.awm_coca.gesim_runtime.save_video", lambda *args, **kwargs: None)

    prepared = PreparedGeSimCondition(
        condition_id="condition",
        sample_dir=str(tmp_path),
        observation=torch.zeros(1, 3, 1, 2, 2),
        cond_to_concat=torch.zeros(2, 1, 2, 2, 2),
        original_trajectory=torch.zeros(1),
        memory_latents=torch.zeros(1, 2, 1, 1, 1),
        prompt_embeds=torch.zeros(1, 1, 1),
        condition_template=object(),
    )

    artifacts = runtime._rollout_batch(
        prepared, seeds=[11, 12], group_dir=tmp_path, prompt="robot arm"
    )

    assert runtime.pipe.end_inputs == ["latents", "denoised"]
    assert len(artifacts) == 2
    for artifact in artifacts:
        assert len(artifact.trajectory) == 3
        assert "denoised" not in artifact.trajectory[0]
        assert [row["denoised"].unique().item() for row in artifact.trajectory[1:]] == [10.0, 11.0]
        assert artifact.final_future_latent.unique().item() == 2.0
