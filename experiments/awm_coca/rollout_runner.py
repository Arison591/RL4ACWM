from __future__ import annotations

import os

from gesim_video_gen_examples.infer_gesim import infer


def run_rollout(
    *,
    config_file: str,
    prep_dir: str,
    output_dir: str,
    seed: int | None = None,
    prompt: str = "best quality, consistent and smooth motion, realistic, clear and distinct.",
    device: str = "cuda",
) -> str:
    trajectory_path = os.path.join(output_dir, "rollout", "trajectory.pt")
    infer(
        config_file=config_file,
        image_root=prep_dir,
        extrinsic_root=prep_dir,
        intrinsic_root=prep_dir,
        action_path=os.path.join(prep_dir, "actions.npy"),
        prompt=prompt,
        save_path=output_dir,
        seed=seed,
        device=device,
        split_views=True,
        # Intermediate denoise videos are disabled: this evaluation needs the
        # latent trajectory, and extra decodes increase peak GPU memory without
        # contributing to reward or credit assignment.
        denoise_interval=0,
        save_trajectory=True,
        trajectory_path=trajectory_path,
    )
    return trajectory_path
