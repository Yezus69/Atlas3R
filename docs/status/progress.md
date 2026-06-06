# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Active branch: `codex/offline-world-builder-v09-map-consistency-optimizer`.
- Offline V0.9 adds a diagnostic map consistency optimizer through
  `offline build-world`.
- `--optimize-map-consistency` implies raw map export and writes optimizer
  artifacts under `optimizer/`.
- Optimized maps are exported under `world_map_optimized/` and remain
  teacher-pseudo, observed-only, unanchored, not measured, and not
  training-quality.
- Training-cache manifests now reference raw and optimized map artifacts while
  remaining `usable_for_training: false`.

## Evidence

- A: `runs/offline_v09_debug_flat_depth_optimizer` decoded the tiny PPM input,
  wrote a raw debug map with 6 fused points and 5 occupied voxels, and marked
  optimizer status `insufficient_witnesses`.
- B: `runs/offline_v09_tum_consistency_optimizer` decoded 60 TUM RGB frames,
  selected 24 keyframes, wrote 24 VGGT and 24 Depth Pro proposals, and exported
  both raw and optimized maps.
- B raw map: 111,550 fused points, 3,641 occupied voxels, 10,716 observed mesh
  triangles.
- B optimized map: 111,808 fused points, 2,435 occupied voxels, 9,848 observed
  mesh triangles.
- B retained point ratio: 1.002312864.
- B relative disagreement mean: 0.087864511 -> 0.046452649.
- B projection residual mean m: 0.027318565 -> 0.009636894.
- C: skipped because no `.mp4`, `.mov`, or `.m4v` was found under `inputs/`,
  `videos/`, `data/`, `datasets/`, or `runs/`.

## Verification

- Passed: `python -m ruff check src tests`.
- Passed: `python -m ruff format src tests`.
- Passed: `python -m ruff format --check src tests`.
- Passed: `python -m mypy src`.
- Passed: `python -m unittest discover -s tests -p "test_*.py"`: 59 tests.
- Passed: `python -m atlas3r --help`.
- Passed: `python -m atlas3r offline --help`.
- Passed: `python -m atlas3r offline build-world --help`.
- Passed: `python -m atlas3r teachers list`.
- Passed: `python -m atlas3r smoke contracts`.
- Passed: `git diff --check` with CRLF warnings only.

## Known Gaps

- VGGT and Depth Pro scales remain unanchored teacher proposals.
- The optimized map improves internal consistency only; it is not measured,
  physically accurate, or training-quality.
- No physical scale anchor, object permanence, final mesh reconstruction, or
  named evaluation report exists.
