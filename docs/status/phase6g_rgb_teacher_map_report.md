# Phase 6G RGB Teacher Map Report

## Scope

Phase 6G adds an offline diagnostic RGB teacher bridge:

- RGB input from `atlas3r_recording`, RGB folders, NPZ/PPM clips, or videos.
- VGGT teacher-pseudo depth, pose, intrinsics, confidence, and optional points.
- Pseudo `DepthObservation` records with `scale_source=rgb_prior`.
- Existing sparse TSDF fusion and Phase 6F observed mesh chunk artifacts.
- Optional eval-only comparison against measured recording depth/pose.

This is not final RGB-only student mapping, realtime mapping, loop closure,
object-aware fusion, hidden-geometry completion, or an accuracy report.

## Evidence Command

```bash
python -m atlas3r runtime map-rgb-teacher --input runs/phase6a_recording_freiburg1_xyz_val/recording --output runs/phase6g_rgb_teacher_freiburg1_xyz_val --teacher vggt --device cuda:0 --max-frames 64 --frame-stride 2 --teacher-window-size 24 --teacher-window-overlap 8 --image-size 518 --voxel-size-m 0.05 --truncation-voxels 3.0 --pixel-stride 12 --export-mesh-chunks --mesh-format ply --rgb-only
```

Environment evidence:

- Teacher: `vggt`.
- Model source: `facebook/VGGT-1B`.
- External package: `vggt 0.0.1` installed from
  `facebookresearch/vggt` commit `a288dd0f14786c93483e45524328726ab7b1b4ce`.
- Device: `cuda:0`.
- No external repository, checkpoint, generated run, or model weight is tracked
  in git.

## Real Run Result

- Input source: `atlas3r_recording_rgb_only`.
- Source sequence: TUM `freiburg1_xyz` validation recording.
- Frames available: 120.
- Frames used: 60 after `--frame-stride 2`.
- Teacher windows: 4.
- Pseudo depth/pose/intrinsics observations: 60 / 60 / 60.
- Pseudo depth valid pixel ratio: `1.0`.
- Sparse map updates: 60.
- Active sparse blocks / voxels: 73 / 6,637.
- Surface points: 2,224.
- Mesh manifest chunks: 60 total, 59 active, 1 deleted.
- Mesh update events: 1,521.
- Mesh vertices / triangles: 24,456 / 12,228.
- Mesh payload bytes: 23,816,040.

Latency diagnostics:

- Teacher inference: `74,697.478 ms`.
- Total pipeline: `92,572.379 ms`.
- Map update p50/p95/max: `45.223 / 48.529 / 49.988 ms`.
- Mesh update p50/p95/max: `74.882 / 92.403 / 164.501 ms`.
- Map+mesh p50/p95/max: `119.970 / 137.883 / 208.140 ms`.

Uncertainty diagnostics:

- Map uncertainty mean/p50/p95/max: `0.056062 / 0.056448 / 0.089964 / 0.099903 m`.
- Mesh uncertainty mean/p50/p95/max: `0.086631 / 0.088395 / 0.097846 / 0.098741 m`.

## Eval-Only Diagnostics

Measured recording depth/pose were not used for mapping. They were read only to
write diagnostic eval sidecars.

- Matched frames: 60.
- Depth alignment: median scale, eval-only.
- Depth AbsRel / RMSE: `0.064570 / 0.255943 m`.
- Depth median scale: `0.980710`.
- Mean valid depth ratio: `0.744961`.
- Pose alignment: Sim3 camera-center, eval-only.
- ATE RMSE / mean / max: `0.085715 / 0.076093 / 0.128713 m`.
- RPE: not implemented in Phase 6G.

## Output Paths

- `rgb_teacher_frames/`
- `rgb_teacher_predictions/`
- `rgb_teacher_recording/`
- `rgb_teacher_summary.json`
- `rgb_teacher_report.md`
- `rgb_teacher_eval.json`
- `rgb_teacher_eval.md`
- `rgb_teacher_eval_camera_centers.csv`
- `live_replay_events.jsonl`
- `live_replay_summary.json`
- `live_replay_report.md`
- `sparse_tsdf/`
- `mesh_chunks/mesh_chunk_manifest.json`
- `mesh_chunks/mesh_chunk_updates.jsonl`
- `mesh_chunks/chunks/*.npz`
- `mesh_chunks/chunks/*.ply`

## Truth Boundary

Mapping flags:

- `diagnostic_only=true`
- `teacher_geometry_used=true`
- `pseudo_depth_used=true`
- `pseudo_pose_used=true`
- `measured_depth_used=false`
- `measured_pose_used=false`
- `observed_only=true`
- `predicted_completion=false`
- `hidden_geometry_measured=false`
- `rgb_only_mapping_ready=false`
- `student_rgb_only_used=false`
- `accuracy_report=false`
- `realtime_claim=false`
- `metric_scale_source=teacher_scale_unverified`

Eval sidecars additionally mark measured depth/pose as available for eval only:
`measured_depth_used_for_eval=true`,
`measured_depth_used_for_mapping=false`,
`measured_pose_used_for_eval=true`, and
`measured_pose_used_for_mapping=false`.

## Verification

- `python -m ruff format src tests`: passed.
- `python -m ruff format --check src tests`: passed.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed.
- `python -m unittest discover -s tests -p "test_*.py"`: passed 249 tests.
- `python -m unittest tests.unit.test_rgb_teacher_mapping tests.unit.test_cli tests.unit.test_live_replay_mesh_chunks tests.unit.test_sparse_tsdf_meshing tests.unit.test_sparse_tsdf_mapper tests.unit.test_external_teachers`: passed 26 tests.
- `python -m compileall -q <new/touched Phase 6G modules>`: passed.
- `git diff --check`: passed; Git emitted CRLF normalization warnings only.

## Known Gaps

- VGGT inference is offline and heavy; no realtime claim exists.
- Monocular teacher scale is unverified unless a later calibration/evaluation
  path proves it.
- Windowed teacher predictions are not loop-closed or globally optimized.
- Sparse TSDF and fallback mesh chunks remain diagnostic CPU paths.
- No object-aware fusion, mesh-quality path, GLB/material export, hidden
  geometry completion, or RGB-only student runtime is implemented.
