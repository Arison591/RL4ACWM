from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from yaml import load, Loader

# 支持从仓库根目录直接执行：
# python experiments/awm_coca/run_eval.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.awm_coca.aggregate_results import write_json, write_rows
from experiments.awm_coca.coca_credit import compute_credit
from experiments.awm_coca.credit_metrics import summarize_credit
from experiments.awm_coca.reward_runner import compute_head_reward
from experiments.awm_coca.rollout_runner import run_rollout
from experiments.awm_coca.trajectory_store import load_trajectory


LOGGER = logging.getLogger("awm_coca.run_eval")


def _parse_seeds(value: str) -> list[int | None]:
    if not value:
        return [None]
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def _write_credit(credit: dict, out_dir: str) -> None:
    write_json({key: value for key, value in credit.items() if key not in {"step_rows", "noise_rows"}}, os.path.join(out_dir, "credit.json"))
    write_rows(credit["step_rows"], os.path.join(out_dir, "credit_steps.csv"))
    write_rows(credit["noise_rows"], os.path.join(out_dir, "credit_noise_levels.csv"))


def _load_eval_config(path: str) -> dict:
    path = path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)
    with open(path, "r", encoding="utf-8") as f:
        return load(f, Loader=Loader) or {}


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"未知 log level: {level}; 可选 DEBUG/INFO/WARNING/ERROR")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AWM-CoCA 推理、head action reward 与 credit assignment 评估")
    parser.add_argument("--eval-config", default="configs/awm_coca_eval.yaml",
                        help="AWM-CoCA 评估配置，包含 rollout/reward/coca 参数")
    parser.add_argument("--config", default=None,
                        help="覆盖 eval-config.rollout.config_file")
    parser.add_argument("--prep-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output-root", default="output/awm_coca_eval")
    parser.add_argument("--samples", default="")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--prompt", default=None, help="覆盖 reward.prompt；默认 robot arm")
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--noise-levels", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                        help="日志级别，默认 INFO")
    parser.add_argument("--skip-rollout", action="store_true", help="只读取已有预测视频和 trajectory")
    args = parser.parse_args()
    _configure_logging(args.log_level)

    eval_config = _load_eval_config(args.eval_config)
    rollout_config = args.config or eval_config.get("rollout", {}).get(
        "config_file", "configs/cosmos_model/acwm_cosmos.yaml"
    )
    reward_config = eval_config.get("reward", {})
    geometry_config = eval_config.get("geometry", {})
    coca_config = eval_config.get("coca", {})
    max_frames = args.max_frames if args.max_frames is not None else int(reward_config.get("max_frames", 49))
    reward_prompt = args.prompt or reward_config.get("prompt", "robot arm")
    confidence = args.confidence if args.confidence is not None else float(reward_config.get("confidence", 0.1))
    action_camera = reward_config.get("action_camera", "head")
    geometry_cameras = list(reward_config.get("geometry_cameras", ["head", "hand_left", "hand_right"]))
    action_metric_weights = dict(reward_config.get("action_metric_weights", {
        "mean_iou": 1 / 3, "af_fdce_ate_norm": 1 / 3, "fdce": 1 / 3,
    }))
    reward_mode = str(reward_config.get("mode", "action")).lower()
    geometry_enabled = bool(geometry_config.get("enabled", False))
    action_weight = float(reward_config.get("action_weight", 1.0))
    geometry_weight = float(reward_config.get("geometry_weight", 1.0))
    window_size = int(coca_config.get("window_size", args.window_size))
    noise_levels = int(coca_config.get("num_training_noise_levels", args.noise_levels))
    temperature = float(coca_config.get("temperature", args.temperature))
    args.config = rollout_config

    samples = [p for p in sorted(Path(args.prep_root).iterdir()) if p.is_dir()]
    if args.samples:
        wanted = {x.strip() for x in args.samples.split(",") if x.strip()}
        samples = [p for p in samples if p.name in wanted]
    if args.limit > 0:
        samples = samples[:args.limit]
    seeds = _parse_seeds(args.seeds)
    os.makedirs(args.output_root, exist_ok=True)
    LOGGER.info(
        "评估开始: samples=%d, seeds=%s, skip_rollout=%s, device=%s",
        len(samples), seeds, args.skip_rollout, args.device,
    )
    LOGGER.info(
        "有效配置: reward_mode=%s, geometry_enabled=%s, action_camera=%s, max_frames=%d, "
        "window_size=%d, noise_levels=%d, temperature=%.4g, action_weights=%s",
        reward_mode, geometry_enabled, action_camera, max_frames, window_size,
        noise_levels, temperature, action_metric_weights,
    )
    write_json({
        **vars(args),
        "eval_config": eval_config,
        "effective_rollout_config": rollout_config,
        "effective_reward_prompt": reward_prompt,
        "effective_confidence": confidence,
        "effective_max_frames": max_frames,
        "effective_action_camera": action_camera,
        "effective_geometry_cameras": geometry_cameras,
        "effective_action_metric_weights": action_metric_weights,
        "effective_reward_mode": reward_mode,
        "effective_geometry_enabled": geometry_enabled,
        "effective_joint_weights": {"action": action_weight, "geometry": geometry_weight},
    }, os.path.join(args.output_root, "run_config.json"))
    manifest = []
    all_summary = []

    for sample_dir in samples:
        sample_id = sample_dir.name
        LOGGER.info("开始 sample=%s", sample_id)
        gt_path = os.path.join(args.gt_root, sample_id, "head_29_frames.mp4")
        sample_pending = []
        for seed in seeds:
            label = "seed_none" if seed is None else f"seed_{seed}"
            out_dir = os.path.join(args.output_root, "samples", sample_id, label)
            pred_dir = os.path.join(out_dir, "videos")
            os.makedirs(pred_dir, exist_ok=True)
            try:
                if not args.skip_rollout:
                    LOGGER.info("[%s/%s] 开始 rollout", sample_id, label)
                    run_rollout(config_file=args.config, prep_dir=str(sample_dir), output_dir=pred_dir, seed=seed, device=args.device)
                    LOGGER.info("[%s/%s] rollout 完成", sample_id, label)
                else:
                    LOGGER.info("[%s/%s] 跳过 rollout，读取已有输出", sample_id, label)
                pred_path = os.path.join(pred_dir, "head.mp4")
                all_camera_videos = {
                    camera: {
                        "gt": os.path.join(args.gt_root, sample_id, f"{camera}_29_frames.mp4"),
                        "pred": os.path.join(pred_dir, f"{camera}.mp4"),
                    }
                    for camera in ("head", "hand_left", "hand_right")
                }
                reward = compute_head_reward(
                    gt_path,
                    pred_path,
                    max_frames=max_frames,
                    prompt=reward_prompt,
                    confidence=confidence,
                    action_metric_weights=action_metric_weights,
                    af_fdce_ate_norm_scale=float(reward_config.get("af_fdce_ate_norm_scale", 0.2)),
                    fdce_scale=float(reward_config.get("fdce_scale", 10.0)),
                    fdce_k=int(reward_config.get("fdce_k", 16)),
                    fdce_visibility_threshold=float(reward_config.get("fdce_visibility_threshold", 0.5)),
                    fdce_min_visible_fraction=float(reward_config.get("fdce_min_visible_fraction", 0.8)),
                    fdce_min_common_frames=int(reward_config.get("fdce_min_common_frames", 1)),
                    fdce_seed=int(reward_config.get("fdce_seed", 0)),
                    prep_dir=str(sample_dir),
                    all_camera_videos=all_camera_videos,
                    geometry_cameras=geometry_cameras,
                    reward_mode=reward_mode,
                    geometry_enabled=geometry_enabled,
                    geometry_future_start=int(geometry_config.get("future_start", 4)),
                    geometry_future_end=int(geometry_config.get("future_end", 28)),
                    geometry_mean_weight=float(geometry_config.get("mean_weight", 0.6)),
                    geometry_worst_weight=float(geometry_config.get("worst_weight", 0.4)),
                    geometry_psnr_center_db=float(geometry_config.get("psnr_center_db", 20.4)),
                    geometry_psnr_temperature_db=float(geometry_config.get("psnr_temperature_db", 1.8)),
                    action_weight=action_weight,
                    geometry_weight=geometry_weight,
                )
                trajectory = load_trajectory(os.path.join(pred_dir, "rollout", "trajectory.pt"))
                write_json(reward, os.path.join(out_dir, "reward.json"))
                if reward.get("total_reward") is None:
                    raise ValueError(f"reward 无效: {reward.get('error')}")
                sample_pending.append({"seed": seed, "out_dir": out_dir, "reward": reward, "trajectory": trajectory})
                LOGGER.info(
                    "[%s/%s] reward=%.6f, action_reward=%s, geometry_reward=%s, "
                    "trajectory_chunks=%d, trajectory_steps=%d",
                    sample_id, label, reward["total_reward"],
                    "none" if reward["action_reward"] is None else f'{reward["action_reward"]:.6f}',
                    "none" if reward["geometry_reward"] is None else f'{reward["geometry_reward"]:.6f}',
                    len(trajectory.get("chunks", [])),
                    sum(max(len(chunk) - 1, 0) for chunk in trajectory.get("chunks", [])),
                )
            except Exception as exc:
                LOGGER.exception("[%s/%s] 评估失败: %s", sample_id, label, exc)
                write_json({"status": "failed", "sample_id": sample_id, "seed": seed, "error": repr(exc)},
                           os.path.join(out_dir, "status.json"))
                manifest.append({"sample_id": sample_id, "seed": seed, "status": "failed", "error": repr(exc)})

        rewards = [item["reward"]["total_reward"] for item in sample_pending]
        for index, item in enumerate(sample_pending):
            reward_value = item["reward"]["total_reward"]
            others = rewards[:index] + rewards[index + 1:]
            baseline = float(np.mean(others)) if others else 0.0
            advantage = float(reward_value - baseline)
            item["reward"]["group_size"] = len(sample_pending)
            item["reward"]["leave_one_out_baseline"] = baseline
            item["reward"]["advantage"] = advantage
            credit = compute_credit(item["trajectory"], reward_value, advantage=advantage,
                                    window_size=window_size, num_training_noise_levels=noise_levels,
                                    temperature=temperature,
                                    credit_source=str(coca_config.get("credit_source", "predicted_x0")))
            LOGGER.info(
                "[%s/seed_%s] credit 完成: reverse_steps=%d, noise_levels=%d, "
                "reward_conservation_error=%.3e",
                sample_id, item["seed"], credit["num_reverse_steps"],
                credit["num_training_noise_levels"], credit["reward_conservation_error"],
            )
            write_json(item["reward"], os.path.join(item["out_dir"], "reward.json"))
            _write_credit(credit, item["out_dir"])
            summary = {"sample_id": sample_id, "seed": item["seed"], "total_reward": reward_value,
                       "baseline": baseline, "advantage": advantage, **summarize_credit(credit)}
            write_json(summary, os.path.join(item["out_dir"], "status.json"))
            all_summary.append(summary)
            manifest.append({"sample_id": sample_id, "seed": item["seed"], "status": "ok", "output": item["out_dir"]})

    write_json({"records": all_summary, "num_ok": len(all_summary)}, os.path.join(args.output_root, "summary.json"))
    write_rows(all_summary, os.path.join(args.output_root, "aggregate.csv"))
    conservation = [row["reward_conservation_error"] for row in all_summary if row.get("reward_conservation_error") is not None]
    write_json({
        "num_records": len(all_summary),
        "reward_conservation_error_mean": float(np.mean(conservation)) if conservation else None,
        "reward_conservation_error_max": float(np.max(conservation)) if conservation else None,
        "intervention_probe": "not_run",
    }, os.path.join(args.output_root, "credit_accuracy.json"))
    with open(os.path.join(args.output_root, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    LOGGER.info("评估完成: ok=%d, manifest=%s", len(all_summary),
                os.path.join(args.output_root, "manifest.jsonl"))


if __name__ == "__main__":
    main()
