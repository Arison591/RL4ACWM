from pathlib import Path

import pytest

from experiments.tempflow_video.run import _initialize_rank_local_sam3_serially


def test_rank_local_sam3_startup_waits_for_predecessor(tmp_path: Path) -> None:
    calls: list[str] = []
    (tmp_path / "rank_0.ready").write_text("ready\n", encoding="utf-8")

    _initialize_rank_local_sam3_serially(
        local_rank=1,
        world_size=4,
        sync_root=tmp_path,
        loader=lambda: calls.append("loaded"),
    )

    assert calls == ["loaded"]
    assert (tmp_path / "rank_1.ready").read_text(encoding="utf-8") == "ready\n"


def test_rank_local_sam3_startup_times_out_without_predecessor(tmp_path: Path) -> None:
    with pytest.raises(TimeoutError, match="rank 0"):
        _initialize_rank_local_sam3_serially(
            local_rank=1,
            world_size=4,
            sync_root=tmp_path,
            loader=lambda: None,
            timeout_seconds=0.0,
        )
