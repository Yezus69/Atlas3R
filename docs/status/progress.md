# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Active branch: `codex/vipe-primary-room-reconstruction`.
- Atlas3R now has `offline import-vipe` for external ViPE run artifacts.
- `offline build-world` can export a finite `world_map_best/` with a local
  `viewer.html` and an optional ground-plane/camera-height near-metric cue.
- ViPE remains outside git at
  `C:/Users/Asav/source/repos/homebrain/external/vipe`.
- Import code is dependency-safe at package import time; OpenEXR/Pillow/PyTorch
  are loaded only inside artifact readers when needed.
- Build-world map outputs remain observed-only teacher consensus, with no
  physical accuracy or measured-geometry claim.

## Room Evidence

- Input:
  `C:/Users/Asav/source/repos/homebrain/data/inbox/room_walk_001/frames`.
- ViPE smoke run:
  `C:/Users/Asav/source/repos/homebrain/external/vipe_runs/room_walk_001_smoke60_final`.
- ViPE main run:
  `C:/Users/Asav/source/repos/homebrain/external/vipe_runs/room_walk_001_main240`.
- Main ViPE command exited `0` and logged `Finished processing frames`.
- Main ViPE artifacts: 240 depth frames, 240 intrinsics, 240 pose rows, and a
  finite dense SLAM map with 168,365 points.
- ViPE `pose/frames.npz` has nonfinite `T_world_camera` rows for all 240
  frames, so Atlas3R does not write a camera trajectory.
- Atlas3R import output: `runs/room_walk_001_vipe_import`.
- Imported ViPE map: 168,365 points, 10,628 occupied voxels, 42,656 observed
  mesh triangles, 0 trajectory poses.
- Imported bbox: about `4.60 m x 3.78 m x 5.24 m`.
- Comparison target:
  `runs/room_walk_001_v11_colmap_witness/world_map_best`.
- Previous V10/V11 map: 1,904,976 points, 2,559 occupied voxels, 5,600
  observed mesh triangles, 88 trajectory poses, bbox about
  `1.72 m x 0.75 m x 0.96 m`.
- ViPE import is less collapsed by bbox heuristic only; this is not a physical
  accuracy claim.
- Latest build-world run:
  `runs/room_walk_001_build_world_near_metric_viewer_final`.
- Latest build-world command used cached VGGT/Depth Pro proposals from
  `runs/room_walk_001_v11_colmap_witness/proposals` and enabled
  `--enable-ground-plane-scale --export-best-world-map-viewer`.
- Latest best map selected `raw_consensus`: 285,516 points, 139,717 occupied
  voxels, 685,404 observed mesh triangles, 88 finite trajectory poses.
- Latest bbox after ground-plane/camera-height cue: about
  `10.01 m x 5.08 m x 6.55 m`.
- Scale status is `near_metric_unanchored`; scale confidence is `very_low`
  because the camera-height cue disagrees strongly with the teacher soft-metric
  prior. This is a near-metric prior, not measured physical scale.
- Latest viewer:
  `runs/room_walk_001_build_world_near_metric_viewer_final/world_map_best/viewer.html`.

## Verification

- Passed: `python -m ruff format src tests`.
- Passed: `python -m ruff check src tests`.
- Passed: `python -m mypy src`.
- Passed: `python -m unittest tests.test_vipe_import`: 3 tests.
- Passed: `python -m unittest tests.test_ground_plane_scale tests.test_best_map_selection tests.test_soft_metric_scale_ledger`: 6 tests.
- Passed: `python -m pytest`: 89 tests.
- Passed: `python -m compileall -q src tests`.
- Passed: `python -m atlas3r offline import-vipe --help`.
- Passed: `python -m atlas3r offline build-world --help`.
- Passed: `python -m atlas3r offline --help`.
- Passed: `python -m atlas3r teachers list`.
- Passed: `python -m atlas3r smoke contracts`.
- Passed: `git diff --check` with line-ending warnings only.
- Passed real import with ViPE venv:
  `python -m atlas3r offline import-vipe --vipe-output ...room_walk_001_main240 --frames ...room_walk_001/frames --output runs/room_walk_001_vipe_import --previous-map runs/room_walk_001_v11_colmap_witness/world_map_best`.
- Passed real build-world run:
  `python -m atlas3r offline build-world --input ...room_walk_001/frames --output runs/room_walk_001_build_world_near_metric_viewer_final --max-frames 240 --keyframe-stride 3 --keyframe-max-count 64 --vggt-proposal-cache ...v11_colmap_witness/proposals --depth-pro-proposal-cache ...v11_colmap_witness/proposals --export-world-map --export-best-world-map --enable-ground-plane-scale --export-best-world-map-viewer --map-depth-source consensus --map-write-occupancy --map-write-observed-mesh --map-point-stride 16 --map-max-points 300000 --map-min-confidence 0.2 --map-max-relative-disagreement 0.35 --map-voxel-size-m 0.05 --write-ply`.
- Passed Edge headless screenshot of the local viewer served at
  `http://127.0.0.1:8765/viewer.html`.
- Passed import safety check: importing `atlas3r` did not load `torch`, `cv2`,
  `open3d`, or `trimesh`.
- Not run: `make test`; `make` is not installed in this PowerShell shell.

## Known Gaps

- ViPE pose export is nonfinite for this run; trajectory import is unavailable.
- The dense SLAM map fallback is inspectable geometry but does not provide
  camera poses.
- Scale remains near-metric and unanchored; the new ground-plane/camera-height
  cue is recorded with `very_low` confidence on this capture.
- No physical accuracy, measured geometry, millimeter accuracy, training-quality
  cache, object fusion, or hidden completion claim exists.
- CP4 object-size anchoring and CP5 dynamic masking were not implemented in this
  task.
