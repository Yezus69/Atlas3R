# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Active branch: `codex/offline-world-builder-v11-colmap-witness`.
- Offline V1.1 adds COLMAP/GLOMAP as dependency-safe classical geometry
  witnesses through `offline build-world`.
- New CLI flags include `--enable-colmap`, `--colmap-exe`,
  `--colmap-camera-model`, `--colmap-matcher`, `--colmap-max-images`,
  `--colmap-image-stride`, `--colmap-use-gpu`, dense/poisson toggles,
  `--enable-glomap`, and replay caches.
- COLMAP sparse text import parses cameras, images, points, colors, errors,
  track lengths, and converts COLMAP world-to-camera poses to Atlas3R
  `T_world_camera`.
- Classical trajectory alignment writes Sim3 metrics and aligned camera/point
  artifacts only when common frames are sufficient.
- Classical map comparison writes nearest-neighbor and bbox agreement against
  raw, optimized, and best maps.
- Optional `world_map_classical_validated/` is selected only when agreement is
  available and the anti-collapse rule keeps at least 50% of points with
  nonzero observed mesh.

## Room Evidence

- Input: `C:/Users/Asav/source/repos/homebrain/data/inbox/room_walk_001/frames`.
- V1.1 run: `runs/room_walk_001_v11_colmap_witness`.
- Decoded 240 frames and selected 64 keyframes.
- VGGT proposals: 88 cameras, 88 depths, 4 windows.
- Depth Pro proposals: 64 cameras, 64 depths.
- COLMAP availability: unavailable; `colmap` was not found on PATH or checked
  common Windows/repo-adjacent paths.
- GLOMAP availability: not enabled; no executable used.
- COLMAP registered images: 0; sparse points: 0.
- Classical alignment: unavailable, common frames 0, RMSE/p95 unavailable.
- Classical map comparison: unavailable because no classical sparse model.
- Best map: selected `optimized`, 1,904,976 points, 2,559 occupied voxels,
  5,600 observed triangles, 88 trajectory poses.
- Physical and training-quality claims remain false.

## Verification

- Passed: `python -m ruff format src tests`.
- Passed: `python -m ruff format --check src tests`.
- Passed: `python -m ruff check src tests`.
- Passed: `python -m mypy src`.
- Passed: `python -m unittest discover -s tests -p "test_*.py"`: 83 tests.
- Passed: `git diff --check` with line-ending warnings only.
- Passed: `python -m atlas3r --help`.
- Passed: `python -m atlas3r offline --help`.
- Passed: `python -m atlas3r offline build-world --help`.
- Passed: `python -m atlas3r teachers list`.
- Passed: `python -m atlas3r smoke contracts`.
- Not run: `make lint`; `make` is not installed in this PowerShell shell. The
  equivalent commands from the Makefile passed.

## Known Gaps

- No COLMAP/GLOMAP reconstruction was produced because COLMAP was unavailable.
- Classical trajectory and sparse-map agreement remain untested on the actual
  room frames until COLMAP/GLOMAP is installed or replayed.
- Scale remains unanchored soft-metric teacher scale.
- No physical scale anchor, object permanence, final mesh reconstruction, or
  named evaluation report exists.
