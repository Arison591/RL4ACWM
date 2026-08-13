from __future__ import annotations

from argparse import Namespace

from experiments.awm_coca.run_train import (
    apply_cli_overrides,
    load_train_config,
)


def test_smoke_group_can_satisfy_its_valid_seed_threshold(tmp_path):
    args = Namespace(
        prep_root=None,
        gt_root=None,
        output_dir=str(tmp_path),
        gesim_config=None,
        checkpoint_root=None,
        rollout_retention=None,
        keep_consumed_rollouts=False,
        group_size=None,
        max_optimizer_steps=None,
        dataset_limit=None,
        num_workers=None,
        reward_workers=None,
        checkpoint_every=None,
        smoke_test=True,
    )

    config = apply_cli_overrides(
        load_train_config("configs/awm_coca_train.yaml"), args
    )

    assert config["rollout"]["group_size"] == 2
    assert config["reward"]["min_valid_seeds_per_group"] == 2
    assert config["max_optimizer_steps"] == 1
