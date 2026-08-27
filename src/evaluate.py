from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    bbox_diagonal,
    read_jsonl,
    sample_key,
    similarity_aligned_error_mm,
)


PCK_THRESHOLDS = (0.05, 0.10)
MIN_VISIBLE_JOINTS = 12
MATCH_NME_THRESHOLD = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("hand_pose_benchmark/results/reference.jsonl"),
    )
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hand_pose_benchmark/results/metrics"),
    )
    return parser.parse_args()


def evaluable_hands(row: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for hand in row["hands"]:
        valid = np.asarray(hand["valid"], dtype=bool)
        if hand.get("bad_reason") or valid.sum() < MIN_VISIBLE_JOINTS:
            continue
        points = np.asarray(hand["landmarks_2d"], dtype=np.float64)
        diagonal = bbox_diagonal(points[valid])
        if diagonal < 16:
            continue
        result.append({**hand, "_valid": valid, "_diagonal": diagonal})
    return result


def pair_cost(reference: dict[str, Any], prediction: dict[str, Any]) -> float:
    valid = reference["_valid"]
    target = np.asarray(reference["landmarks_2d"], dtype=np.float64)
    estimate = np.asarray(prediction["landmarks_2d"], dtype=np.float64)
    if estimate.shape != (21, 2) or not np.isfinite(estimate).all():
        return float("inf")
    return float(
        np.linalg.norm(estimate[valid] - target[valid], axis=1).mean()
        / reference["_diagonal"]
    )


def best_pairs(
    references: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[tuple[int, int, float]]:
    if not references or not predictions:
        return []
    count = min(len(references), len(predictions))
    costs = np.asarray(
        [[pair_cost(gt, pred) for pred in predictions] for gt in references]
    )
    best: tuple[float, list[tuple[int, int, float]]] | None = None
    for gt_indexes in itertools.combinations(range(len(references)), count):
        for pred_indexes in itertools.permutations(range(len(predictions)), count):
            pairs = [
                (gt_index, pred_index, float(costs[gt_index, pred_index]))
                for gt_index, pred_index in zip(gt_indexes, pred_indexes)
            ]
            total = sum(pair[2] for pair in pairs)
            if best is None or total < best[0]:
                best = total, pairs
    assert best is not None
    return [pair for pair in best[1] if pair[2] < MATCH_NME_THRESHOLD]


def empty_accumulator() -> dict[str, Any]:
    return {
        "samples": 0,
        "gt_hands": 0,
        "pred_hands": 0,
        "matched_hands": 0,
        "gt_joints": 0,
        "matched_joints": 0,
        "correct_005": 0,
        "correct_010": 0,
        "matched_correct_005": 0,
        "matched_correct_010": 0,
        "handedness_total": 0,
        "handedness_correct": 0,
        "pixel_errors": [],
        "nme_errors": [],
        "pa_mpjpe_mm": [],
        "runtime_ms": [],
        "landmark_runtime_ms": [],
    }


def add_sample(
    accumulator: dict[str, Any],
    reference_row: dict[str, Any],
    prediction_row: dict[str, Any],
) -> None:
    references = evaluable_hands(reference_row)
    predictions = prediction_row["hands"]
    accumulator["samples"] += 1
    accumulator["runtime_ms"].append(
        float(prediction_row.get("runtime_ms", 0.0))
        + float(prediction_row.get("detector_runtime_ms", 0.0))
    )
    accumulator["landmark_runtime_ms"].append(
        float(prediction_row.get("runtime_ms", 0.0))
    )
    if not references:
        return
    accumulator["gt_hands"] += len(references)
    accumulator["pred_hands"] += len(predictions)
    for reference in references:
        accumulator["gt_joints"] += int(reference["_valid"].sum())

    pairs = best_pairs(references, predictions)
    accumulator["matched_hands"] += len(pairs)
    for gt_index, pred_index, _ in pairs:
        reference = references[gt_index]
        prediction = predictions[pred_index]
        valid = reference["_valid"]
        target = np.asarray(reference["landmarks_2d"], dtype=np.float64)
        estimate = np.asarray(prediction["landmarks_2d"], dtype=np.float64)
        pixel_errors = np.linalg.norm(estimate[valid] - target[valid], axis=1)
        nme_errors = pixel_errors / reference["_diagonal"]
        accumulator["pixel_errors"].extend(pixel_errors.tolist())
        accumulator["nme_errors"].extend(nme_errors.tolist())
        accumulator["matched_joints"] += len(pixel_errors)
        correct_005 = int((nme_errors < PCK_THRESHOLDS[0]).sum())
        correct_010 = int((nme_errors < PCK_THRESHOLDS[1]).sum())
        accumulator["correct_005"] += correct_005
        accumulator["correct_010"] += correct_010
        accumulator["matched_correct_005"] += correct_005
        accumulator["matched_correct_010"] += correct_010
        accumulator["handedness_total"] += 1
        accumulator["handedness_correct"] += int(
            prediction.get("side") == reference["side"]
        )
        if prediction.get("relative_3d"):
            pa_error = similarity_aligned_error_mm(
                np.asarray(prediction["relative_3d"], dtype=np.float64),
                np.asarray(reference["camera_3d"], dtype=np.float64),
            )
            if np.isfinite(pa_error):
                accumulator["pa_mpjpe_mm"].append(pa_error)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def summarize(
    method: str, segment: str, value: str, accumulator: dict[str, Any]
) -> dict[str, Any]:
    matched = accumulator["matched_hands"]
    gt = accumulator["gt_hands"]
    predicted = accumulator["pred_hands"]
    recall = safe_ratio(matched, gt)
    precision = safe_ratio(matched, predicted)
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision + recall) and precision + recall > 0
        else float("nan")
    )
    runtime = np.asarray(accumulator["runtime_ms"], dtype=np.float64)
    landmark_runtime = np.asarray(
        accumulator["landmark_runtime_ms"], dtype=np.float64
    )
    return {
        "method": method,
        "segment": segment,
        "value": value,
        "samples": accumulator["samples"],
        "gt_hands": gt,
        "pred_hands": predicted,
        "matched_hands": matched,
        "hand_recall": recall,
        "hand_precision": precision,
        "hand_f1": f1,
        "e2e_pck_005": safe_ratio(
            accumulator["correct_005"], accumulator["gt_joints"]
        ),
        "e2e_pck_010": safe_ratio(
            accumulator["correct_010"], accumulator["gt_joints"]
        ),
        "matched_pck_005": safe_ratio(
            accumulator["matched_correct_005"], accumulator["matched_joints"]
        ),
        "matched_pck_010": safe_ratio(
            accumulator["matched_correct_010"], accumulator["matched_joints"]
        ),
        "mean_pixel_error": float(np.mean(accumulator["pixel_errors"]))
        if accumulator["pixel_errors"]
        else float("nan"),
        "mean_nme": float(np.mean(accumulator["nme_errors"]))
        if accumulator["nme_errors"]
        else float("nan"),
        "handedness_accuracy": safe_ratio(
            accumulator["handedness_correct"], accumulator["handedness_total"]
        ),
        "pa_mpjpe_mm": float(np.mean(accumulator["pa_mpjpe_mm"]))
        if accumulator["pa_mpjpe_mm"]
        else float("nan"),
        "mean_runtime_ms": float(runtime.mean()) if runtime.size else float("nan"),
        "median_runtime_ms": float(np.median(runtime))
        if runtime.size
        else float("nan"),
        "p95_runtime_ms": float(np.quantile(runtime, 0.95))
        if runtime.size
        else float("nan"),
        "mean_landmark_runtime_ms": float(landmark_runtime.mean())
        if landmark_runtime.size
        else float("nan"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference_quality(reference: list[dict[str, Any]]) -> dict[str, Any]:
    bad_hands = 0
    evaluable = 0
    total = 0
    visible_counts = []
    for row in reference:
        total += len(row["hands"])
        for hand in row["hands"]:
            bad_hands += int(bool(hand.get("bad_reason")))
            visible_counts.append(int(np.asarray(hand["valid"], dtype=bool).sum()))
        evaluable += len(evaluable_hands(row))
    return {
        "sampled_camera_frames": len(reference),
        "reference_hands": total,
        "explicitly_bad_hands": bad_hands,
        "evaluable_hands": evaluable,
        "evaluable_share": safe_ratio(evaluable, total),
        "median_visible_joints": float(np.median(visible_counts)),
        "minimum_visible_joints": MIN_VISIBLE_JOINTS,
        "match_nme_threshold": MATCH_NME_THRESHOLD,
        "pck_thresholds": list(PCK_THRESHOLDS),
    }


def main() -> None:
    args = parse_args()
    reference = read_jsonl(args.reference)
    reference_by_key = {sample_key(row): row for row in reference}
    summaries: list[dict[str, Any]] = []

    for prediction_path in args.predictions:
        predictions = read_jsonl(prediction_path)
        prediction_by_key = {sample_key(row): row for row in predictions}
        missing = set(reference_by_key).difference(prediction_by_key)
        if missing:
            raise ValueError(f"{prediction_path} is missing {len(missing)} samples")
        method = predictions[0]["method"]
        accumulators = {
            "overall": defaultdict(empty_accumulator),
            "task": defaultdict(empty_accumulator),
            "camera": defaultdict(empty_accumulator),
        }
        overall = empty_accumulator()
        for key, reference_row in reference_by_key.items():
            prediction_row = prediction_by_key[key]
            add_sample(overall, reference_row, prediction_row)
            add_sample(
                accumulators["task"][reference_row["task"]],
                reference_row,
                prediction_row,
            )
            add_sample(
                accumulators["camera"][reference_row["camera"]],
                reference_row,
                prediction_row,
            )
        summaries.append(summarize(method, "overall", "all", overall))
        for segment in ("task", "camera"):
            for value, accumulator in sorted(accumulators[segment].items()):
                summaries.append(summarize(method, segment, value, accumulator))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics_all_segments.csv", summaries)
    write_csv(
        args.output_dir / "metrics_overall.csv",
        [row for row in summaries if row["segment"] == "overall"],
    )
    with (args.output_dir / "reference_quality.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(reference_quality(reference), file, indent=2)
    print(json.dumps([r for r in summaries if r["segment"] == "overall"], indent=2))


if __name__ == "__main__":
    main()
