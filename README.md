# Ego hand-pose benchmark

This directory benchmarks maintained open-source hand-pose pipelines against
the paired `EgoStand` / `EgoStand-body` annotations for the locally downloaded
Lightwheel EgoDemo episodes.

The benchmark uses the Standard videos as model inputs because their frames are
exactly aligned with the pose parquet. It borrows only the undistorted camera
intrinsics from each same-UUID `EgoRaw` episode.

Default sampling is every 10th frame in both head cameras. Generated outputs,
downloaded weights, isolated environments, and cloned upstream repositories are
kept under `results/`, `models/`, `envs/`, and `third_party/` respectively.
Only aggregate metrics are committed: EgoDemo-derived frames, reference and
prediction JSONL files, weights, environments, and upstream repositories stay
local.

The runnable baselines are Google MediaPipe Hand Landmarker, OpenMMLab
RTMPose-Hand5 (using either MediaPipe detections or reference boxes), and Meta
Sapiens2-0.4B. Sapiens2 is evaluated as a whole-person 308-keypoint baseline
with a full-frame bounding box because egocentric frames generally do not
contain a conventional full-body person box. Its default benchmark invocation
uses bfloat16 and no optional flip-test ensemble; both choices are exposed as
command-line options.

## Result summary

The deployable recommendation is MediaPipe detection followed by RTMPose-Hand5
landmark refinement. On 830 sampled camera-frames / 1,658 evaluable hands:

| Method | Hand recall | E2E PCK@.05 | E2E PCK@.10 | Median runtime |
|---|---:|---:|---:|---:|
| MediaPipe | 88.7% | 30.2% | 59.1% | 23.7 ms CPU |
| RTMPose + MediaPipe boxes | 88.6% | 43.6% | 74.5% | 34.7 ms CPU+GPU |
| RTMPose + reference boxes (diagnostic only) | 100.0% | 48.5% | 83.2% | 10.9 ms GPU |
| Sapiens2 full-frame | 0.0% | 0.0% | 0.0% | 388.2 ms GPU |

The reference-box row uses ground-truth information and is not deployable. The
Sapiens2 result is a negative domain-transfer result, not a general evaluation
of that whole-person model.

The self-contained result report is at [`report/report.html`](report/report.html).

## Repository contents

- `src/`: reference projection, inference, evaluation, visualization, and report
  artifact builders.
- `episodes.json`: the four evaluated task/episode pairs.
- `results/metrics/`: aggregate benchmark outputs; no images or per-frame poses.
- `report/`: portable report, source artifact, and reviewable SQL queries.
- `patches/`: the two small MMPose Linux ARM64 compatibility edits used in the
  run.

## Reproduce

The script defaults assume this repository remains named `hand_pose_benchmark`
and the commands run from its parent directory. Place the authorized EgoDemo
subset at `EgoDemo_sample/`. Install the Python dependencies required by each
upstream project in isolated environments, then place downloaded model files at
the default paths shown by each script's `--help` output. Model and dataset
licenses are not conveyed by this repository.

The upstream revisions used for the run were:

- MMPose `759b39c13fea6ba094afc1fa932f51dc1b11cbf9`
- Sapiens2 `7e5bae88456ac418ff0e58e74106c9fe192055d4`
- WiLoR `fcb911312a38fa8badd30d9656a167485d61b8f9` (reviewed, not run)

On Linux ARM64, prepare MMPose with:

```bash
git clone https://github.com/open-mmlab/mmpose.git \
  hand_pose_benchmark/third_party/mmpose-main
git -C hand_pose_benchmark/third_party/mmpose-main checkout \
  759b39c13fea6ba094afc1fa932f51dc1b11cbf9
git -C hand_pose_benchmark/third_party/mmpose-main apply \
  ../../patches/mmpose-linux-arm64-lite.patch
```

Run the benchmark with the corresponding environment Python executables:

```bash
# Build the projected EgoDemo reference sample.
python \
  hand_pose_benchmark/src/prepare_reference.py

# Google MediaPipe Hand Landmarker.
PYTHONPATH="$PWD/hand_pose_benchmark/src" \
  hand_pose_benchmark/envs/mediapipe/bin/python \
  hand_pose_benchmark/src/run_mediapipe.py

# OpenMMLab RTMPose-Hand5 using MediaPipe detections.
PYTHONPATH="$PWD/hand_pose_benchmark/third_party/mmpose-main:$PWD/hand_pose_benchmark/src" \
  hand_pose_benchmark/envs/mmpose/bin/python \
  hand_pose_benchmark/src/run_mmpose.py --box-source mediapipe

# Meta Sapiens2 full-frame stress test.
PYTHONPATH="$PWD/hand_pose_benchmark/third_party/sapiens2:$PWD/hand_pose_benchmark/src" \
  hand_pose_benchmark/envs/sapiens2/bin/python \
  hand_pose_benchmark/src/run_sapiens2.py

# Recompute metrics and the representative overlay grid.
python \
  hand_pose_benchmark/src/evaluate.py \
  hand_pose_benchmark/results/predictions_mediapipe.jsonl \
  hand_pose_benchmark/results/predictions_mmpose_rtmpose_mpdet.jsonl \
  hand_pose_benchmark/results/predictions_mmpose_rtmpose_oracle.jsonl \
  hand_pose_benchmark/results/predictions_sapiens2.jsonl
python \
  hand_pose_benchmark/src/render_comparison.py
```

The evaluator reports end-to-end PCK (missed hands count as zero), matched-only
PCK, NME, hand recall/precision/F1, handedness, and runtime. The MediaPipe 3D
number is full-Procrustes hand-shape error only; it is not an absolute camera or
world-coordinate result.

## Due diligence

MediaPipe and MMPose are Apache-2.0 and are the production-oriented choices.
WiLoR and HaWoR require a separately accepted MANO asset license and use
CC BY-NC-ND model terms, so they were researched but not run. Sapiens2 uses a
Meta custom license with usage restrictions; review it before any production or
biometric-processing use.

## Browser validation

The portable report was validated with the Chromium headless-shell version that
matches the report builder's `playwright-core` dependency. The verifier wrapper
hides classic browser scrollbars on Linux ARM64 to match overlay-scrollbar
clients such as macOS browsers; content scrolling remains enabled. Point the
wrapper at another executable with `CHROMIUM_HEADLESS_SHELL` if Playwright's
default cache does not contain it.

```bash
CHROMIUM_EXECUTABLE_PATH="$PWD/hand_pose_benchmark/tools/chromium-headless-verifier" \
  npm run report:deliver -- \
  --input "$PWD/hand_pose_benchmark/report/artifact.json" \
  --output "$PWD/hand_pose_benchmark/report/report.html"
```

The successful check covers 1440 px and 390 px viewports, rendered block/chart/
table counts, non-zero geometry, overflow, external requests, browser errors,
and keyboard access to a representative source menu and dialog.
