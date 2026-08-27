from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


CAMERAS = ("left", "right")
HAND_SIDES = ("left", "right")
HAND_BONES = tuple(
    edge
    for base in (1, 5, 9, 13, 17)
    for edge in zip(
        (0, base, base + 1, base + 2),
        (base, base + 1, base + 2, base + 3),
    )
)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


def nested_array(value: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    """Convert Arrow/Pandas nested object arrays to a numeric ndarray."""
    return np.stack(value).astype(dtype, copy=False)


def quaternion_to_rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Return the local-to-world matrix for a wxyz quaternion."""
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64)
    return np.asarray(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def project_world_points(
    world_xyz: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation_wxyz: np.ndarray,
    intrinsics: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into an undistorted EgoDemo camera image."""
    rotation = quaternion_to_rotation_matrix(camera_rotation_wxyz)
    camera_xyz = (
        np.asarray(world_xyz, dtype=np.float64)
        - np.asarray(camera_position, dtype=np.float64)
    ) @ rotation
    depth = camera_xyz[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = np.column_stack(
            (
                intrinsics["fx"] * camera_xyz[:, 0] / depth + intrinsics["cx"],
                intrinsics["fy"] * camera_xyz[:, 1] / depth + intrinsics["cy"],
            )
        )
    return pixels, camera_xyz


def sample_indices(frame_count: int, stride: int) -> list[int]:
    indexes = list(range(0, frame_count, stride))
    if indexes[-1] != frame_count - 1:
        indexes.append(frame_count - 1)
    return indexes


def sample_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["episode_uuid"], row["camera"], int(row["frame_index"])


def padded_square_bbox(
    points: np.ndarray,
    width: int,
    height: int,
    padding: float = 0.35,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2
    side = max(float((maximum - minimum).max()) * (1 + 2 * padding), 32.0)
    half = side / 2
    return np.asarray(
        [
            max(0.0, center[0] - half),
            max(0.0, center[1] - half),
            min(float(width - 1), center[0] + half),
            min(float(height - 1), center[1] + half),
        ],
        dtype=np.float32,
    )


def bbox_diagonal(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def similarity_aligned_error_mm(
    prediction: np.ndarray, reference: np.ndarray
) -> float:
    """Full Procrustes-aligned per-joint error, in millimetres."""
    source = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    source_norm = np.linalg.norm(source_zero)
    target_norm = np.linalg.norm(target_zero)
    if source_norm < 1e-9 or target_norm < 1e-9:
        return float("nan")
    source_unit = source_zero / source_norm
    target_unit = target_zero / target_norm
    u_matrix, _, vt_matrix = np.linalg.svd(source_unit.T @ target_unit)
    rotation = u_matrix @ vt_matrix
    if np.linalg.det(rotation) < 0:
        u_matrix[:, -1] *= -1
        rotation = u_matrix @ vt_matrix
    aligned = source_unit @ rotation * target_norm + target_center
    return float(np.linalg.norm(aligned - target, axis=1).mean() * 1000.0)
