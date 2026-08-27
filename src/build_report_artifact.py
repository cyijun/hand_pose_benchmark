from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results/metrics"
REPORT_DIR = ROOT / "report"

METHOD_NAMES = {
    "Google MediaPipe Hand Landmarker": "MediaPipe",
    "OpenMMLab RTMPose-m + MediaPipe boxes": "RTMPose + MP boxes",
    "OpenMMLab RTMPose-m + reference boxes": "RTMPose + reference boxes*",
    "Meta Sapiens2-0.4B (full-frame bbox)": "Sapiens2 full frame",
}


def finite_or_none(value: str) -> float | int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def load_metrics() -> list[dict[str, Any]]:
    with (METRICS_DIR / "metrics_all_segments.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    output = []
    numeric = set(rows[0]).difference({"method", "segment", "value"})
    for row in rows:
        clean = {
            key: finite_or_none(value) if key in numeric else value
            for key, value in row.items()
        }
        clean["method_short"] = METHOD_NAMES[clean["method"]]
        clean["run_class"] = (
            "diagnostic_only"
            if "reference boxes" in clean["method"]
            else "runnable_pipeline"
        )
        output.append(clean)
    return output


def project_rows() -> list[dict[str, Any]]:
    return [
        {
            "project": "MediaPipe Hand Landmarker",
            "organization": "Google AI Edge",
            "github_stars": 36700,
            "license": "Apache-2.0",
            "maintenance": "Active; official Python guide updated 2026-08-17",
            "fit": "Hand detection + 21-point 2D + hand-relative 3D",
            "benchmark_status": "Ran — recommended detector / fast baseline",
            "main_risk": "World landmarks are hand-relative, not absolute camera/world pose",
            "official_url": "https://github.com/google-ai-edge/mediapipe",
        },
        {
            "project": "MMPose / RTMPose-Hand5",
            "organization": "OpenMMLab",
            "github_stars": 7900,
            "license": "Apache-2.0",
            "maintenance": "Established ecosystem; tested main commit 2025-08-04",
            "fit": "21-point 2D hand refinement from a crop",
            "benchmark_status": "Ran — recommended landmark refiner",
            "main_risk": "Needs a detector; ARM64 required two small portability edits",
            "official_url": "https://github.com/open-mmlab/mmpose",
        },
        {
            "project": "Sapiens2 Pose 0.4B",
            "organization": "Meta",
            "github_stars": 919,
            "license": "Sapiens2 custom license",
            "maintenance": "Initial release 2026-04; tested commit 2026-05-24",
            "fit": "308 whole-body keypoints; hand points are a subset",
            "benchmark_status": "Ran — reject full-frame ego setup",
            "main_risk": "Not a hand detector; custom license restricts biometric processing",
            "official_url": "https://github.com/facebookresearch/sapiens2",
        },
        {
            "project": "WiLoR",
            "organization": "Imperial College London / SJTU",
            "github_stars": 650,
            "license": "CC BY-NC-ND + MANO + Ultralytics",
            "maintenance": "Official speed update 2026-03; tested repo commit 2026-04-07",
            "fit": "End-to-end 3D hand localization and MANO mesh",
            "benchmark_status": "Not run — MANO account/license asset required",
            "main_risk": "Non-commercial/no-derivatives terms and extra licensed assets",
            "official_url": "https://github.com/rolpotamias/WiLoR",
        },
        {
            "project": "HaWoR",
            "organization": "SJTU / Imperial College London",
            "github_stars": 337,
            "license": "CC BY-NC-ND + MANO",
            "maintenance": "CVPR 2025 Highlight; training code still marked pending",
            "fit": "World-space egocentric video hand reconstruction",
            "benchmark_status": "Not run — heavy stack and licensed MANO asset",
            "main_risk": "DROID-SLAM + Metric3D + old CUDA stack; weak deployability",
            "official_url": "https://github.com/ThunderVVV/HaWoR",
        },
        {
            "project": "Hand Tracking Toolkit",
            "organization": "Meta",
            "github_stars": 74,
            "license": "Apache-2.0; MANO/data terms optional",
            "maintenance": "Official research evaluator with tests",
            "fit": "Metrics, loaders and visualization; no inference model",
            "benchmark_status": "Use as evaluation-design reference only",
            "main_risk": "Not an autolabeling model",
            "official_url": "https://github.com/facebookresearch/hand_tracking_toolkit",
        },
    ]


def source(source_id: str, label: str, query_file: str, **query: Any) -> dict[str, Any]:
    sql = (REPORT_DIR / query_file).read_text(encoding="utf-8")
    return {
        "id": source_id,
        "label": label,
        "path": query_file,
        "query": {
            "engine": "DuckDB 1.x",
            "sql": sql,
            "language": "sql",
            "executed_at": "2026-08-27T16:30:00Z",
            **query,
        },
    }


def main() -> None:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    metrics = load_metrics()
    overall = [row for row in metrics if row["segment"] == "overall"]
    tasks = [row for row in metrics if row["segment"] == "task"]
    overall_by_method = {row["method_short"]: row for row in overall}
    mediapipe = overall_by_method["MediaPipe"]
    rtmpose = overall_by_method["RTMPose + MP boxes"]

    summary = [
        {
            "sampled_camera_frames": 830,
            "evaluable_reference_hands": 1658,
            "rtmpose_e2e_pck_010": rtmpose["e2e_pck_010"],
            "pck_010_gain_vs_mediapipe": (
                rtmpose["e2e_pck_010"] - mediapipe["e2e_pck_010"]
            ),
        }
    ]
    pck_long = []
    for row in overall:
        for threshold, field in (("PCK@0.05", "e2e_pck_005"), ("PCK@0.10", "e2e_pck_010")):
            pck_long.append(
                {
                    "method": row["method_short"],
                    "threshold": threshold,
                    "score": row[field],
                    "hand_recall": row["hand_recall"],
                    "matched_hands": row["matched_hands"],
                    "gt_hands": row["gt_hands"],
                    "run_class": row["run_class"],
                }
            )
    task_long = [
        {
            "task": row["value"],
            "method": row["method_short"],
            "e2e_pck_010": row["e2e_pck_010"],
            "e2e_pck_005": row["e2e_pck_005"],
            "hand_recall": row["hand_recall"],
            "gt_hands": row["gt_hands"],
            "matched_hands": row["matched_hands"],
        }
        for row in tasks
        if row["method_short"] in {"MediaPipe", "RTMPose + MP boxes", "Sapiens2 full frame"}
    ]
    latency = [
        {
            "method": row["method_short"],
            "median_runtime_ms": row["median_runtime_ms"],
            "mean_runtime_ms": row["mean_runtime_ms"],
            "p95_runtime_ms": row["p95_runtime_ms"],
            "landmark_stage_mean_ms": row["mean_landmark_runtime_ms"],
            "device": "CPU" if row["method_short"] == "MediaPipe" else "NVIDIA GB10 GPU",
            "precision": "bfloat16" if "Sapiens2" in row["method_short"] else "default",
            "run_class": row["run_class"],
        }
        for row in overall
        if row["method_short"] != "RTMPose + reference boxes*"
    ]
    projects = project_rows()
    with (METRICS_DIR / "reference_quality.json").open(encoding="utf-8") as file:
        quality_row = json.load(file)
    quality_row["pck_thresholds"] = ", ".join(
        f"{value:.2f}" for value in quality_row["pck_thresholds"]
    )
    quality = [quality_row]

    metrics_source = source(
        "benchmark_metrics",
        "EgoDemo benchmark metric snapshot",
        "queries/benchmark_metrics.sql",
        description="Reads the reviewed evaluator outputs for overall, task, and camera segments.",
        tables_used=["hand_pose_benchmark/results/metrics/metrics_all_segments.csv"],
        filters=[
            "Four downloaded EgoDemo tasks, both head cameras",
            "Every 10th Standard-video frame plus each final frame",
            "Hands with an explicit bad reason, fewer than 12 visible joints, or bbox diagonal below 16 px are excluded",
        ],
        metric_definitions=[
            "End-to-end PCK@t = correctly localized reference joints / all evaluable reference joints; an undetected hand contributes zero correct joints.",
            "Matched PCK@t uses only matched hands; t is normalized by the reference hand bounding-box diagonal.",
            "A reference/prediction pair is matched when mean normalized joint error is below 0.50; pairing is minimum-cost and independent of handedness.",
            "Reported runtime starts after video frame decoding. RTMPose end-to-end runtime adds MediaPipe detection and RTMPose landmark stages.",
        ],
    )
    quality_source = source(
        "reference_quality",
        "EgoDemo reference quality snapshot",
        "queries/reference_quality.sql",
        description="Reads the reference eligibility and explicit bad-frame summary.",
        tables_used=["hand_pose_benchmark/results/metrics/reference_quality.json"],
        filters=["Same deterministic stride-10 camera-frame sample as the benchmark"],
        metric_definitions=[
            "Evaluable share = evaluable reference hands / all sampled reference hands.",
            "Visible joints are projected into the Standard video using the same-UUID Raw undistorted camera intrinsics and per-frame camera pose.",
        ],
    )
    projects_source = source(
        "project_inventory",
        "Official repository due-diligence snapshot",
        "queries/project_inventory.sql",
        description="Curated official-repository metadata checked on 2026-08-27.",
        tables_used=[
            "github.com/google-ai-edge/mediapipe",
            "github.com/open-mmlab/mmpose",
            "github.com/facebookresearch/sapiens2",
            "github.com/rolpotamias/WiLoR",
            "github.com/ThunderVVV/HaWoR",
            "github.com/facebookresearch/hand_tracking_toolkit",
        ],
        filters=["Official repositories only", "Projects relevant to hand coordinates, egocentric reconstruction, or evaluation"],
        metric_definitions=["GitHub stars are point-in-time counts displayed by the official repositories on 2026-08-27."],
    )
    sources = [metrics_source, quality_source, projects_source]

    cards = [
        {
            "id": "sample_frames",
            "description": "Deterministic stride-10 sample across four tasks and both head cameras.",
            "dataset": "summary",
            "sourceId": "benchmark_metrics",
            "metrics": [{"label": "Evaluated camera-frames", "field": "sampled_camera_frames", "format": "number"}],
        },
        {
            "id": "reference_hands",
            "description": "Reference hands that passed bad-frame, visibility, and minimum-size rules.",
            "dataset": "summary",
            "sourceId": "reference_quality",
            "metrics": [{"label": "Evaluable reference hands", "field": "evaluable_reference_hands", "format": "number"}],
        },
        {
            "id": "rtmpose_pck",
            "description": "End-to-end PCK@0.10 for MediaPipe detection followed by RTMPose-Hand5 refinement.",
            "dataset": "summary",
            "sourceId": "benchmark_metrics",
            "metrics": [{"label": "RTMPose end-to-end PCK@0.10", "field": "rtmpose_e2e_pck_010", "format": "percent"}],
        },
        {
            "id": "pck_gain",
            "description": "Absolute PCK@0.10 improvement over MediaPipe landmarks using the same detections.",
            "dataset": "summary",
            "sourceId": "benchmark_metrics",
            "metrics": [{"label": "PCK@0.10 gain vs MediaPipe", "field": "pck_010_gain_vs_mediapipe", "format": "percent", "signed": True}],
        },
    ]
    charts = [
        {
            "id": "overall_pck",
            "title": "End-to-end PCK by method and threshold",
            "subtitle": "RTMPose improves joint placement; the reference-box row is a non-deployable diagnostic upper bound.",
            "intent": "comparison",
            "type": "bar",
            "dataset": "pck_long",
            "sourceId": "benchmark_metrics",
            "encodings": {
                "x": {"field": "method", "type": "nominal", "label": "Method"},
                "y": {"field": "score", "type": "quantitative", "label": "End-to-end PCK", "format": "percent"},
                "color": {"field": "threshold", "type": "nominal", "label": "Threshold"},
                "tooltip": [
                    {"field": "hand_recall", "type": "quantitative", "label": "Hand recall", "format": "percent"},
                    {"field": "matched_hands", "type": "quantitative", "label": "Matched hands"},
                    {"field": "gt_hands", "type": "quantitative", "label": "Reference hands"},
                ],
            },
            "valueFormat": "percent",
            "layout": "full",
            "legend": {"position": "bottom"},
            "settings": {"groupMode": "grouped", "showValues": True},
        },
        {
            "id": "task_pck",
            "title": "End-to-end PCK@0.10 by task",
            "subtitle": "Showerhead scrubbing is the hardest tested sequence; Sapiens2 full-frame does not localize either hand.",
            "intent": "comparison",
            "type": "bar",
            "dataset": "task_long",
            "sourceId": "benchmark_metrics",
            "encodings": {
                "x": {"field": "task", "type": "nominal", "label": "Task"},
                "y": {"field": "e2e_pck_010", "type": "quantitative", "label": "End-to-end PCK@0.10", "format": "percent"},
                "color": {"field": "method", "type": "nominal", "label": "Method"},
                "tooltip": [
                    {"field": "e2e_pck_005", "type": "quantitative", "label": "PCK@0.05", "format": "percent"},
                    {"field": "hand_recall", "type": "quantitative", "label": "Hand recall", "format": "percent"},
                    {"field": "gt_hands", "type": "quantitative", "label": "Reference hands"},
                ],
            },
            "valueFormat": "percent",
            "layout": "full",
            "legend": {"position": "bottom"},
            "settings": {"groupMode": "grouped"},
        },
        {
            "id": "latency",
            "title": "Median per-frame runtime by runnable method",
            "subtitle": "Frame decoding is excluded; MediaPipe ran on CPU, while RTMPose and Sapiens2 used the NVIDIA GB10 GPU.",
            "intent": "comparison",
            "type": "bar",
            "dataset": "latency",
            "sourceId": "benchmark_metrics",
            "encodings": {
                "x": {"field": "method", "type": "nominal", "label": "Method"},
                "y": {"field": "median_runtime_ms", "type": "quantitative", "label": "Median runtime", "unit": "ms"},
                "tooltip": [
                    {"field": "mean_runtime_ms", "type": "quantitative", "label": "Mean runtime", "unit": "ms"},
                    {"field": "p95_runtime_ms", "type": "quantitative", "label": "P95 runtime", "unit": "ms"},
                    {"field": "device", "type": "text", "label": "Device"},
                    {"field": "precision", "type": "text", "label": "Precision"},
                ],
            },
            "valueFormat": "number",
            "unit": "ms",
            "layout": "full",
            "settings": {"showValues": True},
        },
    ]
    tables = [
        {
            "id": "project_shortlist",
            "title": "Project due diligence",
            "subtitle": "Popularity is a trust signal, not a substitute for task fit and license review.",
            "dataset": "projects",
            "sourceId": "project_inventory",
            "defaultSort": {"field": "github_stars", "direction": "desc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "project", "label": "Project", "type": "text"},
                {"field": "organization", "label": "Organization", "type": "text"},
                {"field": "github_stars", "label": "GitHub stars", "format": "compact"},
                {"field": "license", "label": "License", "type": "text"},
                {"field": "maintenance", "label": "Maintenance signal", "type": "text"},
                {"field": "benchmark_status", "label": "This run", "type": "text"},
                {"field": "main_risk", "label": "Main caveat", "type": "text"},
            ],
        },
        {
            "id": "overall_metrics",
            "title": "Overall benchmark detail",
            "subtitle": "Exact evaluated counts and quality/runtime metrics; the starred method uses reference boxes.",
            "dataset": "overall",
            "sourceId": "benchmark_metrics",
            "defaultSort": {"field": "e2e_pck_010", "direction": "desc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "method_short", "label": "Method", "type": "text"},
                {"field": "hand_recall", "label": "Hand recall", "format": "percent"},
                {"field": "hand_precision", "label": "Hand precision", "format": "percent"},
                {"field": "e2e_pck_005", "label": "E2E PCK@.05", "format": "percent"},
                {"field": "e2e_pck_010", "label": "E2E PCK@.10", "format": "percent"},
                {"field": "matched_pck_010", "label": "Matched PCK@.10", "format": "percent"},
                {"field": "mean_pixel_error", "label": "Mean pixel error", "format": "number"},
                {"field": "median_runtime_ms", "label": "Median runtime ms", "format": "number"},
            ],
        },
        {
            "id": "task_metrics",
            "title": "Task-level benchmark detail",
            "subtitle": "Deployable pipelines plus the Sapiens2 negative-domain baseline.",
            "dataset": "task_long",
            "sourceId": "benchmark_metrics",
            "defaultSort": {"field": "e2e_pck_010", "direction": "desc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "task", "label": "Task", "type": "text"},
                {"field": "method", "label": "Method", "type": "text"},
                {"field": "hand_recall", "label": "Hand recall", "format": "percent"},
                {"field": "e2e_pck_005", "label": "E2E PCK@.05", "format": "percent"},
                {"field": "e2e_pck_010", "label": "E2E PCK@.10", "format": "percent"},
                {"field": "gt_hands", "label": "Reference hands", "format": "number"},
                {"field": "matched_hands", "label": "Matched hands", "format": "number"},
            ],
        },
    ]

    title = "EgoDemo 手部坐标开源方案技术评测"
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": "## Technical summary\n\n**建议把 Google MediaPipe 作为手检测与在线跟踪层，把 OpenMMLab RTMPose-Hand5 作为 21 点精修层。** 两者组织可信、社区成熟、Apache-2.0，且已经在当前 ARM64/CUDA 机器和四段 EgoDemo 上跑通。双目三角化、时序滤波和低置信度人工复核应放在模型之后。Meta Sapiens2 可作为可信组织的负向对照，但不适合直接处理只看得到手臂的第一视角全帧。",
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["sample_frames", "reference_hands", "rtmpose_pck", "pck_gain"]},
        {
            "id": "key_findings",
            "type": "markdown",
            "sourceId": "benchmark_metrics",
            "body": "## Key findings\n\n- **RTMPose + MediaPipe boxes 是当前最强可部署组合**：end-to-end PCK@0.10 为 **74.5%**，MediaPipe 原生 landmarks 为 **59.1%**，绝对提升 **15.4 个百分点**。\n- 两者手检召回都约 **88.6%**，说明 RTMPose 主要改善关节落点，无法补回 MediaPipe 漏掉的手。使用 reference boxes 的诊断上界为 **83.2%**，显示检测/裁框仍有约 **8.8 个百分点**空间。\n- Showerhead scrubbing 最难：组合方案 PCK@0.10 为 **65.8%**、召回 **84.5%**；这段更适合做遮挡和运动模糊回归集。\n- Sapiens2 全帧基线在 NME < 0.50 的匹配规则下 **0/1658** 手匹配，median runtime **388.2 ms**。这是明确的任务域不适配，不代表其全身姿态能力无效。",
        },
        {"id": "overall_pck_block", "type": "chart", "chartId": "overall_pck", "layout": "full"},
        {"id": "task_pck_block", "type": "chart", "chartId": "task_pck", "layout": "full"},
        {"id": "latency_block", "type": "chart", "chartId": "latency", "layout": "full"},
        {
            "id": "project_due_diligence",
            "type": "markdown",
            "sourceId": "project_inventory",
            "body": "## Project due diligence\n\nMediaPipe（**36.7k stars**）和 MMPose（**7.9k stars**）同时满足组织可信、社区规模、许可证宽松和任务适配。Sapiens2 来自 Meta 且在 2026 年活跃，但仍较新（**919 stars**），其自定义许可证明确限制 biometric processing，生产使用前必须单独审查。WiLoR 与 HaWoR 更贴近 3D/世界坐标研究问题，但模型是 CC BY-NC-ND，且需要用户自行同意并下载 MANO 资产；本次没有绕过这一授权门槛。",
        },
        {"id": "project_table_block", "type": "table", "tableId": "project_shortlist", "layout": "full"},
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "benchmark_metrics",
            "body": "## Methodology\n\n四个 episode 覆盖 EgoStand 与 EgoStand-body，每段的左右头戴相机 Standard 视频每 10 帧采样一次并包含末帧，共 **830 camera-frames**。Standard 视频与 pose parquet 帧数对齐；同 UUID EgoRaw 只用于提供 undistorted intrinsics。世界坐标手点通过逐帧相机位姿投影到 2D。\n\nPCK 使用 reference hand bounding-box diagonal 归一化。**End-to-end PCK** 的分母包含漏检手，因此比只在成功匹配手上计算的 matched PCK 更适合评价自动标注流水线。NME < 0.50 才建立手匹配，匹配本身不使用左右手标签。视频解码时间不计入推理延迟。",
        },
        {"id": "overall_table_block", "type": "table", "tableId": "overall_metrics", "layout": "full"},
        {"id": "task_table_block", "type": "table", "tableId": "task_metrics", "layout": "full"},
        {
            "id": "reference_quality_section",
            "type": "markdown",
            "sourceId": "reference_quality",
            "body": "## Data quality and validity\n\n采样得到 **1660** 个 reference hands，其中 **2** 个被官方 bad-frame 元数据明确标坏，最终 **1658（99.88%）** 个可评估；median visible joints 为 **21/21**。投影骨架已对四个任务做视频叠加抽查，整体与手部对齐。\n\n这些结果适合作为 pipeline 回归验证，不应被解读成最终公开 benchmark：reference 是数据集提供的三维姿态投影，并非独立人工逐像素复标；相机标定偏差会同时影响 reference；当前也只有四个 episode。下一步应随机抽 100–200 帧做人工盲审，并把完全未见过的 episode 留作冻结测试集。",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": "## Limitations and license caveats\n\n- MediaPipe 此轮使用 IMAGE mode，没有利用 VIDEO mode 的时序跟踪；持续视频部署的召回与抖动需要另测。\n- 采样帧几乎都含两只 reference hands，没有真正的 no-hand negatives；这里的约 99.6% precision 不能外推到连续录像中的误检率。\n- RTMPose 的 deployable 结果依赖 MediaPipe boxes，因此两个阶段不是独立失败源；reference-box 结果只能作为裁框诊断，不能与端到端方法并列宣称。\n- MediaPipe 的 18.6 mm PA-MPJPE 是完整 Procrustes 对齐后的相对手形误差，不是绝对深度或世界坐标误差。\n- Sapiens2 使用 full-frame fallback、bfloat16、无 flip-test；它原生是 top-down whole-person 模型。自定义许可证包含用途限制，不能仅凭“Meta 开源”推断可用于商业生物特征处理。\n- WiLoR/HaWoR 的 MANO、Ultralytics 与模型许可证需要分别接受；这里不提供规避方法，也未把它们标成已跑通。",
        },
        {
            "id": "recommended_pipeline",
            "type": "markdown",
            "body": "## Recommended pipeline\n\n1. **Detect/track:** MediaPipe VIDEO mode 每路相机输出 hand box、handedness、21 点和置信度。\n2. **Refine:** 对每个 padded square crop 跑 RTMPose-Hand5；保留 MediaPipe 原点、RTMPose 精修点和两套 score 以便审计。\n3. **Stereo/world:** 用双目标定做左右视角关联与三角化；用 reprojection error、骨长稳定性和时序速度作为自动质检。MediaPipe world landmarks 只作相对手形先验。\n4. **Human review:** 双目不一致、遮挡、低置信度、手消失/重现附近帧进入人工队列；可在相邻高置信帧间插值，但必须保留来源标记。\n5. **Regression gate:** 先冻结当前四段数据，要求 end-to-end PCK@0.10 不低于 **74%**、hand recall 不低于 **88%**；扩充 unseen episodes 和人工盲审集后再调整门槛。",
        },
        {
            "id": "reproducibility",
            "type": "markdown",
            "body": "## Reproducibility\n\n所有 reference、prediction JSONL、分段 CSV、权重 SHA-256、独立 Python 环境和运行脚本都保存在 `hand_pose_benchmark`。官方权重校验：MediaPipe `fbc2a3…cde1`，RTMPose `b74fb5…6003`，Sapiens2 `f1b7b0…9106`。MMPose 为适配 ARM64/PyTorch 2.11 做了两处有注释的最小改动：跳过未使用的编译扩展 import，并让 CSPNeXt 使用 MMPose 自带 registry。",
        },
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Technical evaluation of maintained open-source hand-coordinate pipelines on four paired Lightwheel EgoDemo episodes.",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [{"id": item["id"], "label": item["label"], "path": item["path"]} for item in sources],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "summary": summary,
                "overall": overall,
                "pck_long": pck_long,
                "task_long": task_long,
                "latency": latency,
                "projects": projects,
                "reference_quality": quality,
            },
        },
        "sources": sources,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "artifact.json"
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
