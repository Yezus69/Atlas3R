# Progress Summary

This file is a rolling current-state summary, not an append-only transcript.
Detailed history belongs in git commits, tests, and older revisions.

## Current State

- Current phase: Phase 5B.1 cleanup complete; Phase 5C is next.
- Latest implementation: `atlas3r.teachers` teacher-signal cache hardening,
  measured TUM forge, explicit raw local ingest, JSON-only inspection, and
  deduplicated teacher-signal CPU TSDF mapping diagnostics.
- Public teacher commands: `atlas3r teachers forge-measured-tum`,
  `ingest-local`, `inspect-signals`, and `map-signals`.
- Teacher-signal caches carry confidence, uncertainty, source clip metadata,
  and measured/pseudo-label truth flags. Raw NPZ ingest maps filenames to source
  clip IDs. `map-signals` deduplicates overlapping frames by `frame_id`.
  Generated caches and runs remain ignored under `data/` and `runs/`.

## Latest Verified Test State

- `python -m ruff format src tests`: passed with 122 files unchanged.
- `python -m ruff format --check src tests`: passed with 122 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 88 source files.
- `python -m unittest discover -s tests -p "test_*.py"`: ran 170 tests and
  passed. A pre-existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git warned changed LF files will convert to CRLF.
- `make test`, `make lint`, and `make typecheck`: not run because `make` is not
  installed in this Windows shell.

## Real-Data Evidence

- Phase 5A TUM clip caches existed locally: train 330 clips, validation 56 clips.
- Phase 5B measured teacher forge wrote ignored caches under `data/tum_rgbd/`:
  train 330 signals and validation 56 signals with configured sigma 0.01 m.
- Validation `inspect-signals --max-clips 32`: RMSE `0.0` m, MAE `0.0` m,
  AbsRel `0.0`, 100% within 1 mm/5 mm/1 cm/5 cm/10 cm, overlap pixels
  2,272,345, mean valid overlap `73.9695638021%`, mean confidence
  `0.7396956380`, pose-center delta `0.0` m.
- Validation `map-signals --max-clips 16 --voxel-size-m 0.05`: 2,661 TSDF
  surface points, unique source frame IDs 338-357, observed coverage estimate
  `0.0373748454`, uncertainty mean `0.0369071745` m, p95 `0.0625409479` m.

## Compact Phase Ledger

- Skeleton: package layout, CLI entry point, Makefile targets, initial tests.
- Phase 0: NumPy contracts, coordinate math, synthetic cube-room session,
  dependency-free inspection, and CPU TSDF reference artifacts.
- Phase 1: dependency-safe teacher adapter stubs, fixture adapter, summaries
  cache, optional arrays, cache inspection, and cache-to-TSDF replay.
- Phase 2: observed MeshChunk/WorldMap sidecars, TSDF output inspection,
  `DepthObservation`, mapper helpers, and deterministic runtime fixture smoke.
- Phase 3: NumPy-only student boundary, RGB frame sources, context-budget
  cleanup, and FramePacket bridges to student/teacher batch contracts.
- Phase 4: optional Torch synthetic training, checkpoint inference bridge,
  TUM RGB-D debug training/eval, block split, v2 model, and TSDF diagnostics.
- Phase 5A: canonical real multi-view TUM clip cache plus tiny temporal
  center-frame depth and relative-translation training path.
- Phase 5B: stable teacher-signal cache plus measured forge, local ingest,
  source-clip inspection, and CPU TSDF map diagnostic bridge.
- Phase 5B.1: removed teacher-signal HTML/SVG previews, fixed raw NPZ filename
  mapping and cache-local source manifest resolution, rejected invalid manifest
  entries, deduplicated overlapping `map-signals` frames, and kept teacher
  source under budget at 1,359 lines with no file over 400.

## Current Known Gaps

- No external teacher runners yet; Depth Pro, VGGT, LingBot-Map, SAM, and DINO
  remain unintegrated.
- No final SMGT transformer, video decoding, live camera runtime,
  object-aware mapping, GLB/PLY export, or benchmark accuracy/performance
  report.
- Optional temporal training from teacher-signal caches is deferred until
  external teacher-signal validation is in place.
