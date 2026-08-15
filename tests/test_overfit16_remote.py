from __future__ import annotations

import hashlib
import os
import subprocess
import tarfile
from argparse import Namespace
from pathlib import Path

import pytest

from experiments.awm_coca.run_train import (
    _prepare_eval_rollout_root,
    _preflight_eval,
    apply_cli_overrides,
    load_train_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
IDS_FILE = REPO_ROOT / "configs" / "awm_coca_overfit16_ids.txt"


def _condition_ids() -> list[str]:
    return [
        line.strip()
        for line in IDS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        prep_root=None,
        gt_root=None,
        output_dir=str(tmp_path / "output"),
        gesim_config=None,
        checkpoint_root=None,
        rollout_retention=None,
        keep_consumed_rollouts=False,
        group_size=16,
        max_optimizer_steps=3,
        dataset_limit=None,
        num_workers=None,
        reward_workers=None,
        checkpoint_every=None,
        eval_prep_root=str(tmp_path / "overfit16" / "prep"),
        eval_every_group_steps=5,
        eval_max_conditions=16,
        eval_seeds_per_condition=8,
        eval_rollout_batch_size=2,
        eval_seed=12345678,
        smoke_test=False,
    )


def _fake_subset(root: Path) -> tuple[Path, Path]:
    prep = root / "prep"
    gt = root / "selected_samples" / "samples"
    for condition_id in _condition_ids():
        condition_prep = prep / condition_id
        condition_gt = gt / condition_id
        condition_prep.mkdir(parents=True)
        condition_gt.mkdir(parents=True)
        (condition_prep / "actions.npy").touch()
        for camera in ("head", "hand_left", "hand_right"):
            (condition_gt / f"{camera}_29_frames.mp4").touch()
    return prep, gt


def test_overfit_eval_cli_overrides_are_resolved(tmp_path: Path) -> None:
    config = apply_cli_overrides(
        load_train_config(REPO_ROOT / "configs" / "awm_coca_train.yaml"),
        _args(tmp_path),
    )

    assert config["rollout"]["group_size"] == 16
    assert config["max_optimizer_steps"] == 3
    assert config["eval"] == {
        "enabled": True,
        "prep_root": str((tmp_path / "overfit16" / "prep").resolve()),
        "validation_mode": "strict",
        "every_group_steps": 5,
        "max_conditions": 16,
        "seeds_per_condition": 8,
        "rollout_batch_size": 2,
        "seed": 12345678,
        "gt_video_template": None,
        "gt_video_templates": None,
    }


def test_eval_batch_size_must_divide_seed_count(tmp_path: Path) -> None:
    config = apply_cli_overrides(
        load_train_config(REPO_ROOT / "configs" / "awm_coca_train.yaml"),
        _args(tmp_path),
    )
    config["eval"]["rollout_batch_size"] = 3

    with pytest.raises(ValueError, match="must be positive and divide"):
        _preflight_eval(config)


def test_eval_cleanup_is_rank_zero_only_and_synchronized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = tmp_path / "eval" / "rollouts" / "stale_group"
    stale.mkdir(parents=True)
    (stale / "group.json").write_text("{}", encoding="utf-8")
    barriers: list[str] = []
    monkeypatch.setattr(
        "experiments.awm_coca.run_train.dist.barrier",
        lambda: barriers.append("barrier"),
    )

    _prepare_eval_rollout_root(tmp_path, rank=3, world_size=8)
    assert stale.is_dir()
    assert barriers == ["barrier"]

    _prepare_eval_rollout_root(tmp_path, rank=0, world_size=8)
    assert not stale.exists()
    assert barriers == ["barrier", "barrier"]


@pytest.mark.parametrize("gpu_count", [4, 8])
def test_overfit_launcher_dry_run_supports_four_and_eight_gpus(
    tmp_path: Path, gpu_count: int
) -> None:
    prep, gt = _fake_subset(tmp_path / "awm_coca_overfit16")
    env = os.environ.copy()
    env.update(
        {
            "DATA_ROOT": str(tmp_path),
            "OVERFIT16_DATA_DIR": str(tmp_path / "awm_coca_overfit16"),
            "MODEL_DIR": str(tmp_path / "models"),
            "OUTPUT_DIR": str(tmp_path / "run"),
            "NPROC_PER_NODE": str(gpu_count),
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(gpu_count))),
            "OVERFIT_DRY_RUN": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "train_overfit16_remote.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"GPUs={gpu_count}" in result.stdout
    assert f"--eval-prep-root {prep}" in result.stdout
    assert f"GT_DIR={gt}" in result.stdout
    assert "--eval-rollout-batch-size 2" in result.stdout


def test_dataset_packager_builds_expected_layout(tmp_path: Path) -> None:
    prep, gt = _fake_subset(tmp_path / "sources")
    archive = tmp_path / "awm_coca_overfit16.tar.gz"
    env = os.environ.copy()
    env.update({"PREP_SOURCE": str(prep), "GT_SOURCE": str(gt)})

    subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "package_awm_coca_overfit16.sh"),
            str(archive),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert archive.is_file()
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    expected_digest = checksum.read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected_digest
    with tarfile.open(archive, "r:gz") as bundle:
        names = set(bundle.getnames())
    for condition_id in _condition_ids():
        assert f"awm_coca_overfit16/prep/{condition_id}/actions.npy" in names
        assert (
            "awm_coca_overfit16/selected_samples/samples/"
            f"{condition_id}/head_29_frames.mp4"
        ) in names
