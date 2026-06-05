# Phase 6H Teacher Stitch Cache Report

## Scope

Phase 6H adds Sim3 overlap stitching for independent VGGT RGB teacher windows
and exports validated temporal teacher caches for future SMGT training. Outputs
remain diagnostic-only teacher-pseudo geometry, not measured mapping, realtime
mapping, hidden-geometry completion, object-aware fusion, or an accuracy report.

## Real Input

- Input: `runs/phase6a_recording_freiburg1_xyz_val/recording`.
- Source: TUM RGB-D `freiburg1_xyz` validation recording.
- Frames available/used: `120 / 120`.
- Measured depth/pose: available for eval-only sidecars, not used for mapping.

## Commands

No-stitch baseline:

```bash
python -m atlas3r runtime map-rgb-teacher --input runs/phase6a_recording_freiburg1_xyz_val/recording --output runs/phase6h_rgb_teacher_nostitch_freiburg1_xyz_val --teacher vggt --device cuda:0 --max-frames 120 --frame-stride 1 --teacher-window-size 12 --teacher-window-overlap 6 --image-size 518 --voxel-size-m 0.05 --truncation-voxels 3.0 --pixel-stride 12 --export-mesh-chunks --mesh-format ply --rgb-only --stitch-windows none
```

Stitched run:

```bash
python -m atlas3r runtime map-rgb-teacher --input runs/phase6a_recording_freiburg1_xyz_val/recording --output runs/phase6h_rgb_teacher_stitched_freiburg1_xyz_val --teacher vggt --device cuda:0 --max-frames 120 --frame-stride 1 --teacher-window-size 12 --teacher-window-overlap 6 --image-size 518 --voxel-size-m 0.05 --truncation-voxels 3.0 --pixel-stride 12 --export-mesh-chunks --mesh-format ply --rgb-only --stitch-windows sim3-overlap --export-teacher-cache --cache-clip-length 8 --cache-clip-stride 4
```

Cache validation:

```bash
python -m atlas3r inspect teacher-temporal-cache --cache runs/phase6h_rgb_teacher_stitched_freiburg1_xyz_val/teacher_temporal_cache
```

## VGGT Settings

- Teacher: `vggt`.
- Model source: `facebook/VGGT-1B`.
- Device: `cuda:0`.
- Image size: `518`.
- Window size/overlap: `12 / 6`.
- Windows: `19`.
- No external repository, checkpoint, generated run, or model weight is tracked
  in git.

## Results

| Metric | No stitch | Stitched |
| --- | ---: | ---: |
| Stitch mode | `none` | `sim3-overlap` |
| Stitch edges accepted/rejected | `0 / 0` | `18 / 0` |
| Pseudo submaps | `19` | `1` |
| Boundary jump mean/p95/max m | `0.035442 / 0.076935 / 0.078969` | `0.066620 / 0.127874 / 0.133376` |
| Overlap center RMSE mean/p95/max m | n/a | `0.066649 / 0.123902 / 0.130932` |
| Sim3 scale min/median/max | n/a | `0.815424 / 1.086051 / 1.260082` |
| Pseudo depth/pose count | `120 / 120` | `120 / 120` |
| Mesh chunks | `59` | `46` |
| Mesh vertices/triangles | `28,980 / 14,490` | `17,184 / 8,592` |
| Teacher temporal cache clips | `0` | `29` |
| Eval-only ATE RMSE m | `0.118310` | `0.094728` |
| Eval-only depth AbsRel/RMSE m | `0.073069 / 0.264572` | `0.073069 / 0.264572` |

Stitched boundary jumps did not improve; they worsened relative to no-stitch.
The diagnostic quality target is still satisfied because eval-only Sim3 camera
center ATE RMSE improved from `0.118310 m` to `0.094728 m`. Measured
depth/pose were not used for mapping or cache labels.

## Cache Validation

- Cache path: `runs/phase6h_rgb_teacher_stitched_freiburg1_xyz_val/teacher_temporal_cache`.
- Format: `atlas3r_teacher_temporal_cache` version `1`.
- Clips: `29`.
- Clip shape: RGB `8x120x160x3`, K `8x3x3`, depth/sigma/confidence
  `8x120x160`.
- Inspector result: passed.
- Dataset bridge: loaded first clip and returned RGB, K, `T_world_camera`,
  depth, sigma, confidence, valid mask, truth flags, and pseudo target weights.
- Default pseudo target weights: depth `0.25`, pose `0.25`.

## Verification

- `python -m ruff format src tests`: passed, 208 files left unchanged.
- `python -m ruff format --check src tests`: passed.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 157 source files.
- `python -m unittest tests.unit.test_rgb_teacher_stitching`: passed 7 tests.
- `python -m unittest tests.unit.test_teacher_temporal_cache`: passed 2 tests.
- `python -m unittest tests.unit.test_rgb_teacher_mapping`: passed 7 tests.
- `python -m unittest tests.unit.test_cli`: passed.
- `python -m unittest discover -s tests -p "test_*.py"`: passed 260 tests.
- `git diff --check`: passed with CRLF normalization warnings only.

## Truth Boundary

Mapping and cache labels preserve:

- `diagnostic_only=true`
- `observed_only=true`
- `predicted_completion=false`
- `hidden_geometry_measured=false`
- `measured_depth_used=false`
- `measured_pose_used=false`
- `pseudo_depth_used=true`
- `pseudo_pose_used=true`
- `teacher_geometry_used=true`
- `student_rgb_only_used=false`
- `rgb_only_mapping_ready=false`
- `realtime_claim=false`
- `accuracy_report=false`

Eval sidecars read measured TUM depth/pose only for diagnostic comparison and
do not alter mapping, mesh, or cache labels.

## Known Gaps

- Boundary jump metrics worsened even though eval-only ATE improved.
- Stitching is sequential adjacent-window Sim3, not loop closure or global pose
  graph optimization.
- Teacher scale remains unverified; no metric accuracy or realtime claim exists.
- Sparse TSDF and mesh chunks remain diagnostic CPU outputs.
- No student training was run in Phase 6H.
