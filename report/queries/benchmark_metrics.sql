-- DuckDB query over the reviewed Python benchmark outputs.
SELECT
  method,
  segment,
  value,
  samples,
  gt_hands,
  pred_hands,
  matched_hands,
  hand_recall,
  hand_precision,
  hand_f1,
  e2e_pck_005,
  e2e_pck_010,
  matched_pck_005,
  matched_pck_010,
  mean_pixel_error,
  mean_nme,
  handedness_accuracy,
  pa_mpjpe_mm,
  mean_runtime_ms,
  median_runtime_ms,
  p95_runtime_ms,
  mean_landmark_runtime_ms
FROM read_csv_auto(
  'hand_pose_benchmark/results/metrics/metrics_all_segments.csv',
  header = true
);
