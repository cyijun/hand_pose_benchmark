from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from common import (
    padded_square_bbox,
    read_jsonl,
    sample_key,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("hand_pose_benchmark/results/reference.jsonl"),
    )
    parser.add_argument(
        "--detector-predictions",
        type=Path,
        default=Path("hand_pose_benchmark/results/predictions_mediapipe.jsonl"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "hand_pose_benchmark/third_party/mmpose-main/configs/"
            "hand_2d_keypoint/rtmpose/hand5/"
            "rtmpose-m_8xb256-210e_hand5-256x256.py"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("hand_pose_benchmark/models/mmpose/rtmpose-m_hand5.pth"),
    )
    parser.add_argument(
        "--box-source", choices=("mediapipe", "reference"), default="mediapipe"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--padding", type=float, default=0.35)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def trusted_torch_load() -> None:
    """Allow the checksum-pinned official OpenMMLab checkpoint on Torch 2.6+."""
    original = torch.load

    def load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = load


def boxes_from_mediapipe(
    prediction: dict[str, Any], width: int, height: int, padding: float
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    boxes = []
    sources = []
    for hand in prediction["hands"]:
        points = np.asarray(hand["landmarks_2d"], dtype=np.float32)
        boxes.append(padded_square_bbox(points, width, height, padding))
        sources.append(hand)
    return boxes, sources


def boxes_from_reference(
    reference: dict[str, Any], padding: float
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    boxes = []
    sources = []
    for hand in reference["hands"]:
        points = np.asarray(hand["landmarks_2d"], dtype=np.float32)
        valid = np.asarray(hand["valid"], dtype=bool)
        if valid.sum() < 8 or hand.get("bad_reason"):
            continue
        boxes.append(
            padded_square_bbox(
                points[valid], reference["width"], reference["height"], padding
            )
        )
        sources.append({"side": hand["side"], "score": 1.0})
    return boxes, sources


def main() -> None:
    args = parse_args()
    trusted_torch_load()
    from mmpose.apis import inference_topdown, init_model

    reference = read_jsonl(args.reference)
    detector_rows = read_jsonl(args.detector_predictions)
    detector = {sample_key(row): row for row in detector_rows}
    if args.limit:
        reference = reference[: args.limit]
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reference:
        by_video[row["video_path"]].append(row)

    model = init_model(
        str(args.config), str(args.checkpoint), device=args.device
    )
    if args.output is None:
        suffix = "mpdet" if args.box_source == "mediapipe" else "oracle"
        args.output = Path(
            f"hand_pose_benchmark/results/predictions_mmpose_rtmpose_{suffix}.jsonl"
        )

    predictions: list[dict[str, Any]] = []
    for video_path, samples in by_video.items():
        sample_by_frame = {int(row["frame_index"]): row for row in samples}
        wanted = set(sample_by_frame)
        capture = cv2.VideoCapture(video_path)
        frame_index = 0
        while wanted:
            ok, image = capture.read()
            if not ok:
                break
            if frame_index not in wanted:
                frame_index += 1
                continue
            row = sample_by_frame[frame_index]
            detector_row = detector[sample_key(row)]
            if args.box_source == "mediapipe":
                boxes, sources = boxes_from_mediapipe(
                    detector_row, row["width"], row["height"], args.padding
                )
            else:
                boxes, sources = boxes_from_reference(row, args.padding)

            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            samples_out = (
                inference_topdown(
                    model,
                    image,
                    bboxes=np.asarray(boxes, dtype=np.float32),
                    bbox_format="xyxy",
                )
                if boxes
                else []
            )
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            runtime_ms = (time.perf_counter() - started) * 1000.0

            hands = []
            for source, output in zip(sources, samples_out):
                instances = output.pred_instances
                keypoints = np.asarray(instances.keypoints)[0]
                scores = np.asarray(instances.keypoint_scores)[0]
                hands.append(
                    {
                        "side": source["side"],
                        "score": float(scores.mean()),
                        "landmarks_2d": keypoints.tolist(),
                        "keypoint_scores": scores.tolist(),
                    }
                )
            method = (
                "OpenMMLab RTMPose-m + MediaPipe boxes"
                if args.box_source == "mediapipe"
                else "OpenMMLab RTMPose-m + reference boxes"
            )
            predictions.append(
                {
                    "method": method,
                    "episode_uuid": row["episode_uuid"],
                    "camera": row["camera"],
                    "frame_index": frame_index,
                    "runtime_ms": runtime_ms,
                    "detector_runtime_ms": (
                        float(detector_row["runtime_ms"])
                        if args.box_source == "mediapipe"
                        else 0.0
                    ),
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
