# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Active branch: `codex/vipe-primary-room-reconstruction`.
- Atlas3R now has `offline import-vipe` for external ViPE run artifacts.
- ViPE remains outside git at
  `C:/Users/Asav/source/repos/homebrain/external/vipe`.
- Import code is dependency-safe at package import time; OpenEXR/Pillow/PyTorch
  are loaded only inside artifact readers when needed.
- ViPE map outputs use the fused map schema and truth boundary
  `teacher_pseudo_vipe_near_metric`.

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

## Verification

- Passed: `python -m ruff format src tests`.
- Passed: `python -m ruff format --check src tests`.
- Passed: `python -m ruff check src tests`.
- Passed: `python -m mypy src`.
- Passed: `python -m unittest tests.test_vipe_import`: 3 tests.
- Passed: `python -m unittest discover -s tests -p "test_*.py"`: 86 tests.
- Passed: `python -m atlas3r offline import-vipe --help`.
- Passed: `python -m atlas3r offline --help`.
- Passed: `python -m atlas3r teachers list`.
- Passed: `python -m atlas3r smoke contracts`.
- Passed: `git diff --check` with line-ending warnings only.
- Passed real import with ViPE venv:
  `python -m atlas3r offline import-vipe --vipe-output ...room_walk_001_main240 --frames ...room_walk_001/frames --output runs/room_walk_001_vipe_import --previous-map runs/room_walk_001_v11_colmap_witness/world_map_best`.
- Not run: `make test`; `make` is not installed in this PowerShell shell.

## Known Gaps

- ViPE pose export is nonfinite for this run; trajectory import is unavailable.
- The dense SLAM map fallback is inspectable geometry but does not provide
  camera poses.
- Scale remains teacher near-metric and unanchored.
- No physical accuracy, measured geometry, millimeter accuracy, training-quality
  cache, object fusion, or hidden completion claim exists.
