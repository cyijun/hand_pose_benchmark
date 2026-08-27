from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import HAND_BONES, read_jsonl, sample_key
from evaluate import best_pairs, evaluable_hands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("hand_pose_benchmark/results/reference.jsonl"),
    )
    parser.add_argument(
        "--mediapipe",
        type=Path,
        default=Path("hand_pose_benchmark/results/predictions_mediapipe.jsonl"),
    )
    parser.add_argument(
        "--rtmpose",
        type=Path,
        default=Path(
            "hand_pose_benchmark/results/predictions_mmpose_rtmpose_mpdet.jsonl"
        ),
    )
    parser.add_argument(
        "--sapiens2",
        type=Path,
        default=Path("hand_pose_benchmark/results/predictions_sapiens2.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hand_pose_benchmark/results/overlays/representative_grid.jpg"),
    )
    return parser.parse_args()


def draw_hand(image: np.ndarray, hand: dict[str, Any], color: tuple[int, int, int]) -> None:
    points = np.asarray(hand["landmarks_2d"], dtype=np.float64)
    valid = np.isfinite(points).all(axis=1)
    if "valid" in hand:
        valid &= np.asarray(hand["valid"], dtype=bool)
    for start, end in HAND_BONES:
        if valid[start] and valid[end]:
            cv2.line(
                image,
                tuple(np.rint(points[start]).astype(int)),
                tuple(np.rint(points[end]).astype(int)),
                color,
                5,
                cv2.LINE_AA,
            )
    for point in points[valid]:
        cv2.circle(image, tuple(np.rint(point).astype(int)), 7, color, -1, cv2.LINE_AA)


def method_panel(
    image: np.ndarray,
    reference: dict[str, Any],
    prediction: dict[str, Any] | None,
    title: str,
) -> np.ndarray:
    canvas = image.copy()
    for hand in evaluable_hands(reference):
        draw_hand(canvas, hand, (255, 255, 0))  # cyan: EgoDemo reference
    if prediction is not None:
        for hand in prediction["hands"]:
            draw_hand(canvas, hand, (255, 0, 255))  # magenta: model
    panel_width = 480
    panel_height = round(canvas.shape[0] * panel_width / canvas.shape[1])
    canvas = cv2.resize(canvas, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(canvas, (0, 0), (panel_width, 42), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        title,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def representative_rows(
    references: list[dict[str, Any]],
    rtmpose_by_key: dict[tuple[str, str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pick the left-camera sample nearest each task's median RTMPose NME."""
    candidates: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for row in references:
        if row["camera"] != "left":
            continue
        refs = evaluable_hands(row)
        pred = rtmpose_by_key[sample_key(row)]
        pairs = best_pairs(refs, pred["hands"])
        if not refs:
            continue
        # Missing hands receive the evaluator's match threshold as a penalty.
        cost = (sum(pair[2] for pair in pairs) + 0.5 * (len(refs) - len(pairs))) / len(
            refs
        )
        candidates[row["task"]].append((cost, row))
    selected = []
    for task in sorted(candidates):
        rows = sorted(candidates[task], key=lambda item: item[0])
        median_cost = float(np.median([item[0] for item in rows]))
        selected.append(min(rows, key=lambda item: abs(item[0] - median_cost))[1])
    return selected


def read_frame(row: dict[str, Any]) -> np.ndarray:
    capture = cv2.VideoCapture(row["video_path"])
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_index"]))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Failed to decode {row['video_path']} frame {row['frame_index']}")
    return frame


def main() -> None:
    args = parse_args()
    references = read_jsonl(args.reference)
    predictions = {
        "MediaPipe": {sample_key(row): row for row in read_jsonl(args.mediapipe)},
        "RTMPose + MP boxes": {
            sample_key(row): row for row in read_jsonl(args.rtmpose)
        },
        "Sapiens2 full frame": {
            sample_key(row): row for row in read_jsonl(args.sapiens2)
        },
    }
    rows = representative_rows(references, predictions["RTMPose + MP boxes"])
    grid_rows = []
    for row in rows:
        key = sample_key(row)
        frame = read_frame(row)
        task_title = f"Reference | {row['task']} | f={row['frame_index']}"
        panels = [method_panel(frame, row, None, task_title)]
        panels.extend(
            method_panel(frame, row, method_predictions[key], method)
            for method, method_predictions in predictions.items()
        )
        grid_rows.append(np.hstack(panels))
    grid = np.vstack(grid_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {args.output} ({grid.shape[1]}x{grid.shape[0]})")
    print("cyan = EgoDemo reference; magenta = model prediction")


if __name__ == "__main__":
    main()
