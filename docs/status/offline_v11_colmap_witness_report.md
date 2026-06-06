# Offline V1.1 COLMAP Witness Report

## Scope

- Branch: `codex/offline-world-builder-v11-colmap-witness`
- Commit: pending before commit
- Actual room input:
  `C:/Users/Asav/source/repos/homebrain/data/inbox/room_walk_001/frames`
- Evidence run: `runs/room_walk_001_v11_colmap_witness`
- Physical accuracy claim: no
- Training-quality claim: no

## Command

```bash
python -m atlas3r offline build-world \
  --input C:/Users/Asav/source/repos/homebrain/data/inbox/room_walk_001/frames \
  --output runs/room_walk_001_v11_colmap_witness \
  --max-frames 240 \
  --keyframe-stride 3 \
  --keyframe-max-count 64 \
  --scale-mode unanchored-soft-metric \
  --enable-vggt \
  --vggt-device cuda:0 \
  --vggt-image-size 384 \
  --vggt-window-size 24 \
  --vggt-window-overlap 8 \
  --vggt-stitch-mode overlap-sim3 \
  --enable-depth-pro \
  --depth-pro-device cuda:0 \
  --depth-pro-checkpoint C:/Users/Asav/source/repos/homebrain/external/ml-depth-pro/checkpoints/depth_pro.pt \
  --enable-colmap \
  --colmap-exe colmap \
  --colmap-matcher sequential \
  --colmap-use-gpu 1 \
  --colmap-camera-model SIMPLE_RADIAL \
  --colmap-max-images 96 \
  --export-world-map \
  --map-depth-source consensus \
  --map-point-stride 8 \
  --map-min-confidence 0.20 \
  --map-max-relative-disagreement 0.35 \
  --map-voxel-size-m 0.05 \
  --map-write-occupancy \
  --map-write-observed-mesh \
  --optimize-map-consistency \
  --optimizer-max-iterations 5 \
  --optimizer-cross-view-pairs 3 \
  --export-optimized-world-map \
  --export-best-world-map \
  --write-ply
```

## Result

- Frames decoded: 240
- Keyframes selected: 64
- VGGT proposals: 88 cameras, 88 depths, 4 windows
- Depth Pro proposals: 64 cameras, 64 depths
- COLMAP availability: unavailable
- GLOMAP availability: not enabled
- COLMAP stage results: no external stage ran because `colmap` was not found
- COLMAP registered image count: 0
- COLMAP sparse point count: 0
- Dense result: not run
- Common frame count with VGGT: 0
- Sim3 alignment: unavailable
- Trajectory RMSE/p95: unavailable
- Sparse point vs map agreement: unavailable
- Best map source before/after classical validation: `optimized` -> `optimized`
- Classical validation effect: none; no sparse model existed
- Best map points: 1,904,976
- Best map occupied voxels: 2,559
- Best map observed triangles: 5,600
- Best map trajectory poses: 88
- Physical accuracy claim: no
- Training-quality claim: no

## Classical Failure

- Status file: `runs/room_walk_001_v11_colmap_witness/classical/classical_status.json`
- Reason: `COLMAP executable not found: colmap`
- Likely reason: COLMAP not installed or not on PATH
- Next debug action: install COLMAP or pass `--colmap-exe` with the full path
- Placeholder artifacts were written for the required classical layout:
  `colmap_run_manifest.json`, `colmap_commands.jsonl`, stdout/stderr tails,
  sparse summary, empty camera/image streams, empty sparse NPZ, and empty PLY.

## Artifacts To Inspect

- `runs/room_walk_001_v11_colmap_witness/world_map_best/fused_points.ply`
- `runs/room_walk_001_v11_colmap_witness/world_map_best/observed_voxel_mesh.ply`
- `runs/room_walk_001_v11_colmap_witness/world_map_best/topdown_preview.svg`
- `runs/room_walk_001_v11_colmap_witness/world_map_best/camera_trajectory.json`
- `runs/room_walk_001_v11_colmap_witness/classical/classical_report.md`
- `runs/room_walk_001_v11_colmap_witness/diagnostics/classical_map_comparison.md`

## Verification

- Passed: `python -m ruff format src tests`
- Passed: `python -m ruff format --check src tests`
- Passed: `python -m ruff check src tests`
- Passed: `python -m mypy src`
- Passed: `python -m unittest discover -s tests -p "test_*.py"`: 83 tests
- Passed: `git diff --check` with line-ending warnings only
- Passed: CLI help/list/smoke commands

## Next Bottleneck

Offline V1.1b should address room-capture pose failure by adding stronger
neural tracks / CoTracker / VGGT track constraints before object work, unless
COLMAP/GLOMAP is installed and V1.1 is rerun first.
