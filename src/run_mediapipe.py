from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from common import read_jsonl, sample_key, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("hand_pose_benchmark/results/reference.jsonl"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("hand_pose_benchmark/models/mediapipe/hand_landmarker.task"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hand_pose_benchmark/results/predictions_mediapipe.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = read_jsonl(args.reference)
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reference:
        by_video[row["video_path"]].append(row)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model.resolve())),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.2,
        min_tracking_confidence=0.2,
    )
    predictions: list[dict[str, Any]] = []
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        for video_path, samples in by_video.items():
            sample_by_frame = {int(row["frame_index"]): row for row in samples}
            wanted = set(sample_by_frame)
            capture = cv2.VideoCapture(video_path)
            frame_index = 0
            while wanted:
                ok, bgr = capture.read()
                if not ok:
                    break
                if frame_index not in wanted:
                    frame_index += 1
                    continue
                row = sample_by_frame[frame_index]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                started = time.perf_counter()
                result = landmarker.detect(image)
                runtime_ms = (time.perf_counter() - started) * 1000.0
                hands = []
                for index, landmarks in enumerate(result.hand_landmarks):
                    category = result.handedness[index][0]
                    hands.append(
                        {
                            "side": category.category_name.lower(),
                            "score": float(category.score),
                            "landmarks_2d": [
                                [float(point.x * row["width"]), float(point.y * row["height"])]
                                for point in landmarks
                            ],
                            "relative_3d": [
                                [float(point.x), float(point.y), float(point.z)]
                                for point in result.hand_world_landmarks[index]
                            ],
                        }
                    )
                predictions.append(
                    {
                        "method": "Google MediaPipe Hand Landmarker",
                        "episode_uuid": row["episode_uuid"],
                        "camera": row["camera"],
                        "frame_index": frame_index,
                        "runtime_ms": runtime_ms,
                        "hands": hands,
                    }
                )
                wanted.remove(frame_index)
                frame_index += 1
            capture.release()
            if wanted:
                raise RuntimeError(f"Could not decode frames {sorted(wanted)} from {video_path}")
            print(f"processed {len(samples)} samples from {Path(video_path).name}")

    predictions.sort(key=sample_key)
    write_jsonl(args.output, predictions)
    print(f"wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
