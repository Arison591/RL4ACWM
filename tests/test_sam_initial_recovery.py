import numpy as np

from experiments.action_following.sam_tracking import _recover_initial_masks


class _FakeSAM:
    def add_prompt(self, inference_state, frame_idx, text_str):
        mask = np.zeros((1, 6, 8), dtype=bool)
        mask[:, 2:5, 3:7] = True
        return frame_idx, {"out_binary_masks": mask}

    def propagate_in_video(self, inference_state, start_frame_idx, reverse):
        assert start_frame_idx == 2
        assert reverse is True
        for frame_idx in (1, 0):
            mask = np.zeros((1, 6, 8), dtype=bool)
            mask[:, 2:5, 3:7] = True
            yield frame_idx, {"out_binary_masks": mask}


def test_later_detection_is_back_propagated_to_frame_zero():
    masks = np.zeros((4, 6, 8), dtype=bool)
    masks[2:, 2:5, 3:7] = True

    recovered, prompt_frame = _recover_initial_masks(
        _FakeSAM(),
        {},
        masks,
        prompt="robot arm",
    )

    assert prompt_frame == 2
    assert recovered[:3].all(axis=0).sum() == 12
    assert recovered[0].sum() == 12


def test_empty_sequence_is_not_fabricated():
    masks = np.zeros((4, 6, 8), dtype=bool)

    recovered, prompt_frame = _recover_initial_masks(
        _FakeSAM(),
        {},
        masks,
        prompt="robot arm",
    )

    assert prompt_frame is None
    assert not recovered.any()
