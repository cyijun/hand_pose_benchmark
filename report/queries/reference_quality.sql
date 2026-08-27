-- DuckDB query over the quality summary emitted by src/evaluate.py.
SELECT
  sampled_camera_frames,
  reference_hands,
  explicitly_bad_hands,
  evaluable_hands,
  evaluable_share,
  median_visible_joints,
  minimum_visible_joints,
  match_nme_threshold,
  pck_thresholds
FROM read_json_auto(
  'hand_pose_benchmark/results/metrics/reference_quality.json'
);
