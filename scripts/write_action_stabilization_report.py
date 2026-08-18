#!/usr/bin/env python3
"""Produce the evidence-based Action RL stabilization report from run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row]
    return mean(values) if values else None


def _fmt(value: float | int | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--noise-floor", required=True)
    parser.add_argument("--paired-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    groups = _jsonl(run_dir / "rollout_groups.jsonl")
    steps = _jsonl(run_dir / "optimizer_steps.jsonl")
    gradients = _jsonl(run_dir / "group_gradient_diagnostics.jsonl")
    noise = json.loads(Path(args.noise_floor).read_text(encoding="utf-8"))
    paired = json.loads(Path(args.paired_report).read_text(encoding="utf-8"))
    valid = [row for row in groups if not row.get("excluded_from_optimizer", False)]
    invalid = [row for row in groups if row.get("excluded_from_optimizer", False)]

    total_groups = len(groups)
    effective_ratio = len(valid) / total_groups if total_groups else 0.0
    command_contribution = _mean(valid, "mean_abs_weighted_command")
    fdce_contribution = _mean(valid, "mean_abs_weighted_fdce")
    iou_contribution = _mean(valid, "mean_abs_weighted_iou")
    command_rank = _mean(valid, "spearman_command_total_advantage")
    fdce_rank = _mean(valid, "spearman_fdce_total_advantage")
    iou_rank = _mean(valid, "spearman_iou_total_advantage")
    coherence = _mean(steps, "window_policy_gradient_coherence")
    policy_kl_cosine = _mean(gradients, "policy_kl_grad_cosine")
    previous_cosine = _mean(gradients, "policy_grad_cosine_previous_group")
    max_step = max((int(row["optimizer_step"]) for row in steps), default=0)

    metrics = paired["metrics"]
    error = metrics["combined_raw_command_error"]
    reward = metrics["total_reward"]
    command_reward = metrics["command_train_reward"]
    error_improved = float(error["mean_difference"]) < 0.0
    command_dominates = (
        command_contribution is not None
        and fdce_contribution is not None
        and iou_contribution is not None
        and command_contribution > fdce_contribution
        and command_contribution > iou_contribution
        and command_rank is not None
        and fdce_rank is not None
        and iou_rank is not None
        and command_rank > fdce_rank
        and command_rank > iou_rank
    )

    floors = noise["recommended_component_noise_floors"]
    lines = [
        "# Action RL Stabilization Report",
        "",
        "## Reward Bug And Current Implementation",
        "",
        "The historical reward merged two arms into one centroid. The corrected evaluator matches left and right YOLO tracks to their corresponding command trajectories independently. Training uses `-combined_raw_command_error`; the compatibility metric `final_command_component` remains an evaluation field.",
        "",
        "## Command Dynamic Range And Repeatability",
        "",
        f"- Repeatability sample: {noise['rollouts']} saved rollouts across {noise.get('condition_count', 'n/a')} conditions, {noise['repeats']} scoring repeats each.",
        f"- P95 noise floors: command={_fmt(floors['command'])}, FDCE={_fmt(floors['fdce'])}, IoU={_fmt(floors['iou'])}.",
        "- YOLO tracker is reset before and after every scoring pass; SAM3 state is per-call and CoWTracker is stateless for this path.",
        "",
        "## Component-Wise Advantage And Gating",
        "",
        "Each high-is-good component is population-z-scored inside its branch group. The final advantage is `0.7 A_command + 0.2 m_fdce A_fdce + 0.1 m_iou A_iou`; no second z-score or advantage clipping is applied. Command validity requires two arms, full coverage, and group standard deviation above its noise floor. Invalid command groups are skipped rather than being driven by FDCE/IoU.",
        f"- Attempts: {total_groups}; valid command groups: {len(valid)} ({effective_ratio:.1%}); skipped: {len(invalid)}.",
        f"- Mean absolute weighted contribution: command={_fmt(command_contribution)}, FDCE={_fmt(fdce_contribution)}, IoU={_fmt(iou_contribution)}.",
        f"- Mean Spearman with total advantage: command={_fmt(command_rank)}, FDCE={_fmt(fdce_rank)}, IoU={_fmt(iou_rank)}.",
        f"- Command dominated the observed training ranking: {'yes' if command_dominates else 'no'}.",
        "",
        "## Multi-Group Gradient And Training",
        "",
        "The correction uses TempFlow branch-only sampling with branch factor 12, four valid groups accumulated under one policy version, one inner epoch, and reference KL. It is a single-pass on-policy GRPO-style update; PPO clipping is retained as a diagnostic, not claimed as active.",
        f"- Optimizer steps: {max_step}; completed update records: {len(steps)}.",
        f"- Mean four-group policy-gradient coherence: {_fmt(coherence)}.",
        f"- Mean policy/KL gradient cosine: {_fmt(policy_kl_cosine)}; mean cosine with preceding group: {_fmt(previous_cosine)}.",
        "",
        "## Fixed 16x8 Paired Evaluation",
        "",
        f"- Paired samples: {paired['pairs']}.",
        f"- Raw command error (lower is better), final-base mean difference: {_fmt(error['mean_difference'])}; median={_fmt(error['median_difference'])}; win/tie/loss for lower error={error['losses']}/{error['ties']}/{error['wins']}.",
        f"- Training-direction command reward (-error), final-base mean difference: {_fmt(command_reward['mean_difference'])}; win/tie/loss={command_reward['wins']}/{command_reward['ties']}/{command_reward['losses']}.",
        f"- Total reward, final-base mean difference: {_fmt(reward['mean_difference'])}; win/tie/loss={reward['wins']}/{reward['ties']}/{reward['losses']}.",
        f"- FDCE mean difference: {_fmt(metrics['fdce']['mean_difference'])}; IoU mean difference: {_fmt(metrics['mean_iou']['mean_difference'])}.",
        "",
        "## Final Conclusion",
        "",
        f"1. Command reward is {'an effective within-group ranking signal' if command_dominates and effective_ratio > 0 else 'not yet established as a reliable within-group ranking signal'} in this bounded run.",
        f"2. FDCE/IoU {'did not dominate' if command_dominates else 'still may dominate'} the observed advantage.",
        f"3. Four-group accumulation {'has diagnostic evidence' if coherence is not None else 'has no completed diagnostic evidence'} for gradient coherence.",
        f"4. Action following {'improved' if error_improved else 'did not improve over base'} under the required fixed paired evaluation.",
        f"5. {'Further action RL is justified only after reviewing this paired result.' if error_improved else 'Do not claim action-following improvement or tune small optimizer parameters further.'}",
        "6. Noise-aware weighting remains disabled in this correction; restoring it would require a separate, explicitly scoped experiment after this action-signal result.",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
