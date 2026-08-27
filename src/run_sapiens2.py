from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from common import read_jsonl, sample_key, write_jsonl


# Sapiens2's 308-keypoint schema stores a 21-joint left hand followed by a
# 21-joint right hand. The order in each slice is wrist, then four joints for
# thumb/index/middle/ring/pinky, matching EgoDemo's hand topology.
HAND_KEYPOINTS = {"left": np.arange(91, 112), "right": np.arange(112, 133)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("hand_pose_benchmark/results/reference.jsonl"),
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("hand_pose_benchmark/third_party/sapiens2"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("hand_pose_benchmark/models/sapiens2/model.safetensors"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hand_pose_benchmark/results/predictions_sapiens2.jsonl"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision", choices=("float32", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--flip-test",
        action="store_true",
        help="Enable Sapiens2's optional test-time horizontal-flip ensemble.",
    )
    return parser.parse_args()


def initialise_model(args: argparse.Namespace) -> Any:
    # Imported lazily so that --help remains usable outside the Sapiens2 env.
    from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
    from sapiens.pose.models import init_model

    config = (
        args.repo
        / "sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/"
        "sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py"
    )
    model = init_model(str(config), str(args.checkpoint), device=args.device)
    metainfo = args.repo / "sapiens/pose/configs/_base_/keypoints308.py"
    model.pose_metainfo = parse_pose_metainfo(dict(from_file=str(metainfo)))
    codec_config = dict(model.cfg.codec)
    codec_type = codec_config.pop("type")
    if codec_type != "UDPHeatmap":
        raise ValueError(f"Expected UDPHeatmap codec, got {codec_type}")
    model.codec = UDPHeatmap(**codec_config)
    model.eval()
    return model


def infer_full_frame(
    model: Any,
    image: np.ndarray,
    precision: str = "bfloat16",
    flip_test: bool = False,
) -> tuple[list[dict[str, Any]], float]:
    """Run the official top-down pipeline using the full image as its one bbox.

    Sapiens2 is a person-pose model rather than a hand detector. EgoDemo views do
    not usually contain a complete person, so a full-frame crop is the least
    assumption-heavy runnable fallback and is reported explicitly in the method
    name. Runtime includes preprocessing, inference, flip-test, and decoding.
    """
    height, width = image.shape[:2]
    data_info = {
        "img": image,
        "bbox": np.asarray([[0, 0, width - 1, height - 1]], dtype=np.float32),
        "bbox_score": np.ones(1, dtype=np.float32),
    }
    torch.cuda.synchronize()
    started = time.perf_counter()
    data = model.pipeline(data_info)
    data = model.data_preprocessor(data)
    inputs = data["inputs"]
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[precision]
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=dtype, enabled=precision != "float32"
    ):
        prediction = model(inputs)
        if flip_test:
            flipped = model(inputs.flip(-1)).flip(-1)
            flipped = flipped[:, model.pose_metainfo["flip_indices"]]
            prediction = (prediction + flipped) / 2.0
    prediction = prediction.float().cpu().numpy()
    keypoints, scores = model.codec.decode(prediction[0])
    metadata = data["data_samples"]["meta"]
    keypoints = (
        keypoints / metadata["input_size"] * metadata["bbox_scale"]
        + metadata["bbox_center"]
        - 0.5 * metadata["bbox_scale"]
    )[0]
    scores = scores[0]
    torch.cuda.synchronize()
    runtime_ms = (time.perf_counter() - started) * 1000.0

    hands = []
    for side, indexes in HAND_KEYPOINTS.items():
        hands.append(
            {
                "side": side,
                "score": float(np.mean(scores[indexes])),
                "landmarks_2d": keypoints[indexes].astype(float).tolist(),
                "keypoint_scores": scores[indexes].astype(float).tolist(),
            }
        )
    return hands, runtime_ms


def main() -> None:
    args = parse_args()
    reference = read_jsonl(args.reference)
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reference:
        by_video[row["video_path"]].append(row)

    model = initialise_model(args)
    predictions: list[dict[str, Any]] = []
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
            hands, runtime_ms = infer_full_frame(
                model, bgr, precision=args.precision, flip_test=args.flip_test
            )
            predictions.append(
                {
                    "method": "Meta Sapiens2-0.4B (full-frame bbox)",
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
        print(f"processed {len(samples)} samples from {Path(video_path).name}", flush=True)

    predictions.sort(key=sample_key)
    write_jsonl(args.output, predictions)
    print(f"wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
