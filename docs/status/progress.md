# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Active branch: `codex/offline-world-builder-v10-room-walk-soft-metric`.
- Offline V1.0 adds no-anchor soft-metric room-walk export through
  `offline build-world`.
- New CLI flags: `--scale-mode unanchored-soft-metric` and
  `--export-best-world-map`.
- Runs now write JPG/EXIF metadata summaries, soft-metric scale hypotheses,
  updated scale ledgers, conservative best-map selection, top-down preview,
  inspection instructions, and room diagnostics.
- Maps remain unanchored teacher-consensus artifacts, not measured geometry,
  not physical ground truth, and not training-quality.

## Room Evidence

- Input: `C:/Users/Asav/source/repos/homebrain/data/inbox/room_walk_001/frames`.
- Baseline current run: `runs/room_walk_001_v10_baseline_current`.
- Final V1.0 run: `runs/room_walk_001_v10_soft_metric`.
- Baseline decoded 180 frames, selected 48 keyframes, wrote 1,403,335 raw
  points, 3,138 occupied voxels, 7,332 observed triangles, and optimizer status
  `improved`.
- Final decoded 240 frames, selected 64 keyframes, wrote 88 VGGT camera/depth
  proposals and 64 Depth Pro camera/depth proposals.
- Final raw map: 1,638,747 points, 3,846 occupied voxels, 8,184 triangles.
- Final optimized map: 2,000,000 points, 3,245 occupied voxels, 7,620
  triangles.
- Final best map: selected `optimized`, 1,904,976 points, 2,559 occupied
  voxels, 5,600 observed triangles, 88 trajectory poses, one kept component.
- Final optimizer improved relative disagreement mean 0.308152169 ->
  0.0612177588 and projection residual mean 0.0498260930 m -> 0.0196512938 m.
- Scale mode: `unanchored_soft_metric`; confidence: `medium`; physical and
  training-quality claims: false.
- Inspect first: `world_map_best/fused_points.ply`,
  `world_map_best/observed_voxel_mesh.ply`,
  `world_map_best/topdown_preview.svg`,
  `world_map_best/camera_trajectory.json`, and
  `world_map_best/map_quality.md`.
- Longer 480-frame/96-keyframe room run skipped because the 240-frame run
  already produced required inspectable artifacts and used substantial GPU/RAM
  time; doubling it risked a long OOM-prone rerun without changing the V1.0
  acceptance proof.

## Regression

- Cached TUM regression: `runs/offline_v10_tum_regression`.
- Decoded 60 frames, selected 24 keyframes, replayed 24 VGGT and 24 Depth Pro
  proposals, selected optimized best map.
- Best TUM map: 96,603 points, 1,630 occupied voxels, 6,220 observed triangles.
- TUM was used only as a regression, not the V1.0 success proof.

## Verification

- Passed: `python -m ruff format src tests`.
- Passed: `python -m ruff format --check src tests`.
- Passed: `python -m ruff check src tests`.
- Passed: `python -m mypy src`.
- Passed: `python -m unittest discover -s tests -p "test_*.py"`: 66 tests.
- Passed: `git diff --check` with line-ending warnings only.
- Passed: `python -m atlas3r --help`.
- Passed: `python -m atlas3r offline --help`.
- Passed: `python -m atlas3r offline build-world --help`.
- Passed: `python -m atlas3r teachers list`.
- Passed: `python -m atlas3r smoke contracts`.

## Known Gaps

- Scale remains unanchored soft-metric teacher scale.
- Room EXIF/focal metadata was unavailable, so intrinsics remain proposal-only.
- Teacher agreement on the room frames is weak before optimization; confidence
  is medium, not high.
- No physical scale anchor, object permanence, final mesh reconstruction, or
  named evaluation report exists.
