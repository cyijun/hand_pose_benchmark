from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from common import (
    CAMERAS,
    HAND_SIDES,
    nested_array,
    project_world_points,
    read_json,
    sample_indices,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("EgoDemo_sample"))
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path("hand_pose_benchmark/episodes.json"),
    )
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hand_pose_benchmark/results/reference.jsonl"),
    )
    return parser.parse_args()


def hand_bad_reasons(path: Path) -> dict[int, dict[str, str | None]]:
    details = read_json(path).get("hand_bad_frame_details", [])
    result: dict[int, dict[str, str | None]] = {}
    for detail in details:
        result[int(detail["frame"])] = {
            "left": detail.get("left_bad_reason") or detail.get("bad_reason"),
            "right": detail.get("right_bad_reason") or detail.get("bad_reason"),
        }
    return result


def build_rows(
    dataset_root: Path, episode: dict[str, str], stride: int
) -> list[dict[str, Any]]:
    standard_root = (
        dataset_root
        / episode["variant"]
        / "lerobot"
        / episode["standard_task"]
        / episode["uuid"]
    )
    raw_root = (
        dataset_root
        / "EgoRaw"
        / "mp4"
        / episode["raw_task"]
        / episode["uuid"]
    )
    parquet = standard_root / "data/chunk-000/file-000.parquet"
    table = pd.read_parquet(parquet)
    bad_reasons = hand_bad_reasons(standard_root / "bad_frame_ratio.json")
    selected = sample_indices(len(table), stride)
    rows: list[dict[str, Any]] = []

    for camera in CAMERAS:
        video_path = (
            standard_root
            / f"videos/observation.images.head_{camera}/chunk-000/file-000.mp4"
        )
        capture = cv2.VideoCapture(str(video_path))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if video_frames != len(table):
            raise ValueError(
                f"Frame mismatch for {video_path}: video={video_frames}, poses={len(table)}"
            )

        parameters = read_json(
            raw_root
            / f"head_{camera}_camera/head_{camera}_camera_params.json"
        )
        intrinsics = parameters["undistorted_intrinsics"]
        for frame_index in selected:
            source = table.iloc[frame_index]
            camera_position = np.asarray(
                source[f"observation.state.head_{camera}_camera_position"],
                dtype=np.float64,
            )
            camera_rotation = np.asarray(
                source[f"observation.state.head_{camera}_camera_rotation"],
                dtype=np.float64,
            )
            hands: list[dict[str, Any]] = []
            for hand_side in HAND_SIDES:
                world = nested_array(
                    source[f"observation.state.hand_{hand_side}_world"],
                    dtype=np.float64,
                )
                pixels, camera_xyz = project_world_points(
                    world, camera_position, camera_rotation, intrinsics
                )
                valid = (
                    (camera_xyz[:, 2] > 0.03)
                    & np.isfinite(pixels).all(axis=1)
                    & (pixels[:, 0] >= 0)
                    & (pixels[:, 0] < width)
                    & (pixels[:, 1] >= 0)
                    & (pixels[:, 1] < height)
                )
                hands.append(
                    {
                        "side": hand_side,
                        "world_3d": world.tolist(),
                        "camera_3d": camera_xyz.tolist(),
                        "landmarks_2d": pixels.tolist(),
                        "valid": valid.tolist(),
                        "bad_reason": bad_reasons.get(frame_index, {}).get(hand_side),
                    }
                )
            rows.append(
                {
                    "episode_uuid": episode["uuid"],
                    "task": episode["task"],
                    "variant": episode["variant"],
                    "camera": camera,
                    "frame_index": frame_index,
                    "timestamp": float(source["timestamp"]),
                    "video_path": str(video_path.resolve()),
                    "width": width,
                    "height": height,
                    "fps": video_fps,
                    "intrinsics": intrinsics,
                    "hands": hands,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    episodes = read_json(args.episodes)
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        episode_rows = build_rows(args.dataset_root, episode, args.stride)
        rows.extend(episode_rows)
        print(
            f"prepared {episode['task']}: {len(episode_rows)} camera-frame samples"
        )
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} samples to {args.output}")


if __name__ == "__main__":
    main()
