# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Current phase: Phase 6E live replay scheduler and camera adapter boundary
  completed on branch `codex/phase6e-live-replay-scheduler`.
- Latest implementation: `runtime live-replay-recording` replays measured
  `atlas3r_recording` folders with deterministic simulated pacing, bounded
  capture/map queues, explicit drop and keyframe reasons, separate pose updates,
  measured-depth/pose-only sparse TSDF map updates, JSONL events, summary, and
  Markdown report.
- Added dependency-safe runtime capture adapter boundary with `OpenCVCameraAdapter`
  lazy `cv2` handling and `ReplayRecordingAdapter` metadata streaming.
- Existing `runtime fuse-recording --mode incremental --backend
  cpu-persistent|cpu-rebuild|cpu-sparse` remains covered by regression tests.

## Latest Verified Test State

- `python -m ruff format src tests`: passed; 183 files unchanged.
- `python -m ruff format --check src tests`: passed; 183 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 138 source files.
- Focused tests passed: `tests.unit.test_live_replay_scheduler`,
  `tests.unit.test_capture_adapters`, `tests.unit.test_recording_runtime`, and
  `tests.unit.test_cli`.
- `python -m unittest discover -s tests -p "test_*.py"`: passed 233 tests. The
  pre-existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git warned changed LF files will convert to CRLF.
- `make` commands were not run because `make` is not installed in this Windows
  shell.

## Real-Data Evidence

- Command:
  `python -m atlas3r runtime live-replay-recording --recording runs/phase6a_recording_freiburg1_xyz_val/recording --output runs/phase6e_live_replay_freiburg1_xyz_val --target-fps 30 --max-frames 120 --mapper-backend cpu-sparse --map-keyframe-stride 1 --max-capture-queue 4 --max-map-queue 2 --drop-policy oldest --voxel-size-m 0.05 --truncation-voxels 3.0 --export-point-cloud`.
- Result: 120 measured TUM `freiburg1_xyz` validation frames seen/emitted, 120
  pose updates, 120 selected keyframes, 120 map updates, no frame/keyframe
  drops, max capture/map queue depth `1/1`, and simulated pacing `true`.
- Sparse state: 142 active blocks, 10,165 active voxels, approximate state bytes
  `585,040`, 2,950 observed surface points, and `surface_points.ply` exported
  under ignored `runs/`.
- Map update latency ms p50/p95/max: `63.936 / 68.0898 / 73.9971`.
- Observation load latency ms p50/p95/max: `15.44545 / 17.04904 / 51.3417`.
- Full evidence: `docs/status/phase6e_live_replay_scheduler_report.md`.

## Compact Phase Ledger

- Phase 0-4: contracts, synthetic correctness, CPU TSDF diagnostics, teacher
  adapter/cache boundaries, NumPy student boundary, optional Torch training MVP,
  checkpoint bridge, and TUM RGB-D debug train/eval.
- Phase 5A-5H: TUM clip caches, teacher-signal caches, external teacher
  runners, temporal measured/pseudo training, student-map runtime diagnostics,
  multi-sequence training, and dependency-safe VGGT runner scaffolding.
- Phase 6A-6D: measured `atlas3r_recording`, sensor-folder importer, dense
  persistent/rebuild incremental TSDF, sparse block TSDF, and sparse memory
  diagnostics.
- Phase 6E: bounded live replay scheduler plus runtime capture adapter boundary.

## Current Known Gaps

- CPU sparse map updates are still too slow for a realtime mapper; p95 was about
  68.1 ms on the measured replay.
- Phase 6E exports observed surface points and sparse state, not live triangle
  mesh chunks or game-engine streaming assets.
- No RGB-only mapping readiness, object-aware fusion, loop closure, accelerated
  mapper, real camera hardware CI exercise, or benchmark accuracy report exists.
