from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CAMERAS = ("head", "hand_left", "hand_right")
HISTORY_INDICES = (0, 1, 2, 3)


@dataclass(frozen=True)
class ConditionEntry:
    index: int
    condition_id: str
    relative_path: str


@dataclass
class RawConditionSample:
    condition_id: str
    sample_dir: str
    history_images: dict[str, torch.Tensor]
    actions: torch.Tensor
    extrinsics: dict[str, torch.Tensor]
    intrinsics: dict[str, torch.Tensor]
    original_sizes: dict[str, tuple[int, int]]


def discover_samples(
    prep_root: str | Path,
    *,
    include_samples: Iterable[str] = (),
    exclude_samples: Iterable[str] = (),
    limit: int = 0,
) -> list[Path]:
    root = Path(prep_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"prep root does not exist: {root}")
    include = set(include_samples)
    exclude = set(exclude_samples)
    samples = [path for path in sorted(root.iterdir()) if path.is_dir() and not path.name.startswith(".")]
    if include:
        missing = include.difference(path.name for path in samples)
        if missing:
            raise ValueError(f"requested samples are missing: {sorted(missing)}")
        samples = [path for path in samples if path.name in include]
    samples = [path for path in samples if path.name not in exclude]
    if limit > 0:
        samples = samples[:limit]
    if not samples:
        raise ValueError(f"no condition directories found under {root}")
    return samples


def _required_paths(sample_dir: Path, cameras: Sequence[str], history_indices: Sequence[int]) -> list[Path]:
    paths = [sample_dir / "actions.npy"]
    for camera in cameras:
        paths.extend((sample_dir / f"extrinsic_{camera}.npy", sample_dir / f"intrinsic_{camera}.npy"))
        paths.extend(sample_dir / f"{camera}_color" / f"{index}.png" for index in history_indices)
    return paths


def validate_sample(
    sample_dir: str | Path,
    *,
    cameras: Sequence[str] = CAMERAS,
    history_indices: Sequence[int] = HISTORY_INDICES,
    expected_actions_shape: tuple[int, int] = (29, 16),
) -> dict[str, Any]:
    directory = Path(sample_dir)
    missing = [str(path) for path in _required_paths(directory, cameras, history_indices) if not path.is_file()]
    if missing:
        raise ValueError(f"{directory.name}: missing required files: {missing}")

    actions = np.load(directory / "actions.npy", mmap_mode="r")
    if tuple(actions.shape) != expected_actions_shape:
        raise ValueError(f"{directory.name}: actions shape {actions.shape}, expected {expected_actions_shape}")
    if not np.isfinite(actions).all():
        raise ValueError(f"{directory.name}: actions contain non-finite values")

    camera_stats: dict[str, Any] = {}
    for camera in cameras:
        extrinsic = np.load(directory / f"extrinsic_{camera}.npy", mmap_mode="r")
        intrinsic = np.load(directory / f"intrinsic_{camera}.npy", mmap_mode="r")
        if extrinsic.ndim != 3 or extrinsic.shape[0] < expected_actions_shape[0] or extrinsic.shape[-2:] != (4, 4):
            raise ValueError(f"{directory.name}: invalid {camera} extrinsic shape {extrinsic.shape}")
        if intrinsic.shape != (3, 3):
            raise ValueError(f"{directory.name}: invalid {camera} intrinsic shape {intrinsic.shape}")
        if not np.isfinite(extrinsic[: expected_actions_shape[0]]).all() or not np.isfinite(intrinsic).all():
            raise ValueError(f"{directory.name}: {camera} camera arrays contain non-finite values")
        if float(intrinsic[0, 0]) <= 0 or float(intrinsic[1, 1]) <= 0:
            raise ValueError(f"{directory.name}: {camera} intrinsic focal length must be positive")

        sizes = []
        for index in history_indices:
            image_path = directory / f"{camera}_color" / f"{index}.png"
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                sizes.append((rgb.height, rgb.width))
        camera_stats[camera] = {
            "extrinsic_shape": list(extrinsic.shape),
            "intrinsic_shape": list(intrinsic.shape),
            "image_sizes": [list(size) for size in sizes],
        }
    return {"condition_id": directory.name, "actions_shape": list(actions.shape), "cameras": camera_stats}


def build_manifest(
    prep_root: str | Path,
    *,
    include_samples: Iterable[str] = (),
    exclude_samples: Iterable[str] = (),
    limit: int = 0,
    validation_mode: str = "strict",
    cameras: Sequence[str] = CAMERAS,
    history_indices: Sequence[int] = HISTORY_INDICES,
    expected_actions_shape: tuple[int, int] = (29, 16),
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if validation_mode not in {"strict", "skip_invalid"}:
        raise ValueError("validation_mode must be strict or skip_invalid")
    root = Path(prep_root).expanduser().resolve()
    candidates = discover_samples(root, include_samples=include_samples, exclude_samples=exclude_samples, limit=limit)
    entries = []
    invalid = []
    validation = []
    for path in candidates:
        try:
            validation.append(validate_sample(path, cameras=cameras, history_indices=history_indices,
                                              expected_actions_shape=expected_actions_shape))
            entries.append({"index": len(entries), "condition_id": path.name, "relative_path": path.name})
        except Exception as exc:
            invalid.append({"condition_id": path.name, "error": str(exc)})
            if validation_mode == "strict":
                raise
    if not entries:
        raise ValueError("dataset has no valid samples")
    manifest = {
        "prep_root": str(root),
        "camera_order": list(cameras),
        "history_indices": list(history_indices),
        "expected_actions_shape": list(expected_actions_shape),
        "num_samples": len(entries),
        "samples": entries,
        "validation": validation,
    }
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return manifest, invalid


def write_manifest(manifest: dict[str, Any], invalid: Sequence[dict[str, str]], output_dir: str | Path) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    with (target / "invalid_samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in invalid:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class PrepConditionDataset(Dataset[RawConditionSample]):
    def __init__(self, manifest: dict[str, Any], *, load_images: bool = True) -> None:
        self.manifest = manifest
        self.root = Path(manifest["prep_root"])
        self.entries = [ConditionEntry(**entry) for entry in manifest["samples"]]
        self.cameras = tuple(manifest["camera_order"])
        self.history_indices = tuple(manifest["history_indices"])
        self.load_images = load_images

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> RawConditionSample:
        entry = self.entries[index]
        directory = self.root / entry.relative_path
        images: dict[str, torch.Tensor] = {}
        original_sizes: dict[str, tuple[int, int]] = {}
        if self.load_images:
            for camera in self.cameras:
                frames = []
                for frame_index in self.history_indices:
                    with Image.open(directory / f"{camera}_color" / f"{frame_index}.png") as image:
                        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
                    frames.append(torch.from_numpy(array).permute(2, 0, 1))
                images[camera] = torch.stack(frames)
                original_sizes[camera] = (int(frames[0].shape[1]), int(frames[0].shape[2]))
        actions = torch.from_numpy(np.load(directory / "actions.npy").astype(np.float32, copy=False))
        extrinsics = {
            camera: torch.from_numpy(np.load(directory / f"extrinsic_{camera}.npy").astype(np.float32, copy=False))[: actions.shape[0]]
            for camera in self.cameras
        }
        intrinsics = {
            camera: torch.from_numpy(np.load(directory / f"intrinsic_{camera}.npy").astype(np.float32, copy=False))
            for camera in self.cameras
        }
        return RawConditionSample(entry.condition_id, str(directory), images, actions, extrinsics, intrinsics, original_sizes)


def collate_single_condition(samples: Sequence[RawConditionSample]) -> RawConditionSample:
    if len(samples) != 1:
        raise ValueError(f"condition batch size must be 1, got {len(samples)}")
    return samples[0]
