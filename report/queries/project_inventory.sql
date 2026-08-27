-- Curated 2026-08-27 snapshot of official repository metadata and due diligence.
SELECT *
FROM (
  VALUES
    ('MediaPipe Hand Landmarker', 'Google AI Edge', 36700, 'Apache-2.0',
     'run_recommended', '2D landmarks, handedness, hand-relative 3D',
     'https://github.com/google-ai-edge/mediapipe'),
    ('MMPose / RTMPose-Hand5', 'OpenMMLab', 7900, 'Apache-2.0',
     'run_recommended', '2D hand landmarks; requires a hand box',
     'https://github.com/open-mmlab/mmpose'),
    ('Sapiens2 Pose 0.4B', 'Meta', 919, 'Sapiens2 custom license',
     'run_rejected_for_ego_full_frame', '308 whole-body keypoints; not a hand detector',
     'https://github.com/facebookresearch/sapiens2'),
    ('WiLoR', 'Imperial College London / SJTU', 650, 'CC BY-NC-ND + MANO',
     'not_run_license_asset_required', 'End-to-end 3D hand mesh and localization',
     'https://github.com/rolpotamias/WiLoR'),
    ('HaWoR', 'SJTU / Imperial College London', 337, 'CC BY-NC-ND + MANO',
     'not_run_heavy_license_asset_required', 'World-space egocentric hand motion',
     'https://github.com/ThunderVVV/HaWoR'),
    ('Hand Tracking Toolkit', 'Meta', 74, 'Apache-2.0 + data/model licenses',
     'evaluation_utility_only', 'Egocentric metrics, data loading and visualization',
     'https://github.com/facebookresearch/hand_tracking_toolkit')
) AS projects(project, organization, github_stars, license, status, task_fit, official_url);
