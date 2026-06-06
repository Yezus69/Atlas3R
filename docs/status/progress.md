# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Active branch: `codex/offline-world-builder-v08-fused-map-output`.
- Offline V0.8 exports the first persistent fused world-map artifacts through
  `offline build-world --export-world-map`.
- The map exporter uses VGGT `T_world_camera` plus diagnostic VGGT/Depth-Pro
  consensus depth when both witnesses exist, and VGGT depth for VGGT-only runs.
- Depth Pro without VGGT pose writes an explicit missing-global-pose failure and
  does not create a fake global map.
- `world_map/` now contains manifest, camera trajectory, fused points NPZ/PLY,
  sparse occupancy NPZ/metadata, optional observed voxel mesh, and map-quality
  JSON/Markdown.
- Training-cache manifests reference fused-map artifacts but remain
  `usable_for_training: false`.

## Evidence

- A: `runs/offline_v08_debug_flat_depth_map` decoded 6 PPM frames, selected 6
  keyframes, wrote 6 fused debug points, 5 occupied voxels, 96 mesh vertices,
  48 triangles, and `inspectable_map_available: true`.
- B: `runs/offline_v08_vggt_world_map` decoded 60 TUM RGB frames, selected 24
  keyframes, wrote 24 VGGT camera/depth proposals, 76,440 fused points, 2,313
  occupied voxels, 21,248 mesh vertices, 10,624 triangles, and
  `inspectable_map_available: true`.
- C: `runs/offline_v08_consensus_world_map` decoded 60 TUM RGB frames, selected
  24 keyframes, wrote 24 VGGT and 24 Depth Pro proposals, 111,550 fused points,
  3,641 occupied voxels, 21,432 mesh vertices, 10,716 triangles, and
  `inspectable_map_available: true`.
- D: skipped because no `.mp4` or `.mov` was found under `runs/`, `data/`,
  `datasets/`, `videos/`, or `inputs/`.

## Verification

- Passed: `python -m ruff format src tests`.
- Passed: `python -m ruff format --check src tests`.
- Passed: `python -m ruff check src tests`.
- Passed: `python -m mypy src`.
- Passed: `python -m unittest discover -s tests -p "test_*.py"`: 55 tests.
- Passed: `python -m atlas3r --help`, `python -m atlas3r offline --help`,
  `python -m atlas3r offline build-world --help`,
  `python -m atlas3r teachers list`, and
  `python -m atlas3r smoke contracts`.
- Passed: `git diff --check` with Windows line-ending warnings only.

## Known Gaps

- VGGT and Depth Pro scales are unanchored teacher proposals.
- The fused map is not optimized, measured, physically accurate, or
  training-quality.
- No physical scale anchor, render-repair optimizer, object permanence, final
  mesh reconstruction, or named evaluation report exists.
