# Offline V1.0 Room Walk Soft-Metric Report

## Scope

- Branch: `codex/offline-world-builder-v10-room-walk-soft-metric`
- Actual room input:
  `C:/Users/Asav/source/repos/homebrain/data/inbox/room_walk_001/frames`
- Baseline run: `runs/room_walk_001_v10_baseline_current`
- Final run: `runs/room_walk_001_v10_soft_metric`
- TUM regression: `runs/offline_v10_tum_regression`

## Baseline Current

- Decoded frames: 180
- Keyframes selected: 48
- VGGT proposals: 64 cameras, 64 depths
- Depth Pro proposals: 48 cameras, 48 depths
- Raw map: 1,403,335 points, 3,138 occupied voxels, 7,332 triangles
- Optimized map: 2,000,000 points, 2,320 occupied voxels, 6,108 triangles
- Optimizer relative mean: 0.2699733377 -> 0.0372168422
- Optimizer projection residual mean m: 0.0384554639 -> 0.0170048196

## Final Room Result

- Decoded frames: 240
- Keyframes selected: 64
- VGGT proposals: 88 cameras, 88 depths, 4 windows
- Depth Pro proposals: 64 cameras, 64 depths
- Raw map: 1,638,747 points, 3,846 occupied voxels, 8,184 triangles
- Optimized map: 2,000,000 points, 3,245 occupied voxels, 7,620 triangles
- Best map source: optimized
- Best map: 1,904,976 points, 2,559 occupied voxels, 5,600 triangles
- Best trajectory poses: 88
- Best bbox size m: `[1.7188382149, 0.7523568869, 0.9621474147]`
- Best cleanup retained ratio: 0.952488
- Best connected components kept: 1

## Optimizer Metrics

- Relative diff mean: 0.3081521690 -> 0.0612177588
- Relative diff p95: 0.5382905602 -> 0.2680483773
- Absolute diff mean m: 0.3597153723 -> 0.0490078256
- Absolute diff p95 m: 0.7220951915 -> 0.2222838998
- Projection residual mean m: 0.0498260930 -> 0.0196512938
- Projection residual p95 m: 0.1623124629 -> 0.0718048155
- Projection count: 1,046,496 -> 1,034,399
- Optimizer status: improved

## Artifacts

- `runs/room_walk_001_v10_soft_metric/world_map_best/fused_points.ply`
- `runs/room_walk_001_v10_soft_metric/world_map_best/observed_voxel_mesh.ply`
- `runs/room_walk_001_v10_soft_metric/world_map_best/topdown_preview.svg`
- `runs/room_walk_001_v10_soft_metric/world_map_best/camera_trajectory.json`
- `runs/room_walk_001_v10_soft_metric/world_map_best/map_quality.md`
- `runs/room_walk_001_v10_soft_metric/room_walk_001_report.md`
- `runs/room_walk_001_v10_soft_metric/world/soft_metric_scale_ledger.json`

## Scale And Claims

- Scale mode: `unanchored_soft_metric`
- Scale status: `soft_metric_unanchored`
- Scale source: `depth_pro_vggt_soft_metric_prior`
- Scale confidence: medium
- EXIF focal metadata: unavailable
- Physical accuracy claim: false
- Training-quality claim: false

## Longer Run

Skipped. The 240-frame/64-keyframe room run produced the required inspectable
V1.0 artifacts and used substantial GPU/RAM time. A 480-frame/96-keyframe run
would double the expensive teacher/optimizer workload and risk OOM without
changing the V1.0 acceptance proof.

## Regression

- Cached TUM run decoded 60 frames and selected 24 keyframes.
- Replayed 24 VGGT and 24 Depth Pro proposals.
- Selected optimized best map with 96,603 points, 1,630 occupied voxels, and
  6,220 observed triangles.
- This was regression coverage only, not the main proof.

## Verification

- Passed: `python -m ruff format src tests`
- Passed: `python -m ruff format --check src tests`
- Passed: `python -m ruff check src tests`
- Passed: `python -m mypy src`
- Passed: `python -m unittest discover -s tests -p "test_*.py"`: 66 tests
- Passed: `git diff --check` with line-ending warnings only
- Passed: `python -m atlas3r --help`
- Passed: `python -m atlas3r offline --help`
- Passed: `python -m atlas3r offline build-world --help`
- Passed: `python -m atlas3r teachers list`
- Passed: `python -m atlas3r smoke contracts`
