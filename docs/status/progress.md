# Progress Summary

This file is a rolling current-state summary, not an append-only transcript.
Detailed history belongs in git commits, tests, and older revisions.

## Current State

- Current phase: Phase 4B complete; `docs/status/next_task.md` now points to
  Phase 4C coherent checkpoint inference sequence smoke.
- Latest implementation: dependency-safe Phase 4A checkpoint inference bridge
  that loads `checkpoint_last.pt`, preserves truth-boundary flags, runs
  `TinyDepthPoseNet` on `StudentClipInput` or ordered `FramePacket` inputs, and
  converts predictions to validated `DepthObservation`.
- `atlas3r smoke checkpoint-tsdf --checkpoint <checkpoint_last.pt> --output <dir>
  [--input <clip.npz>]` writes predicted CPU TSDF artifacts, `prediction_sample.npz`,
  and, for deterministic synthetic input, target TSDF plus predicted-vs-target
  metrics and HTML/SVG preview.
- Predicted observations and TSDF metadata carry depth uncertainty, confidence,
  coordinate frame, RGB-prior metric scale source, source frame IDs, and the
  Phase 4A synthetic-only/not-accuracy-report truth boundary.

## Latest Verified Test State

- `python -m ruff format src tests`: ran; 3 files reformatted.
- `python -m ruff format --check src tests`: passed with 91 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 63 source files.
- `python -m unittest discover -s tests -p 'test_*.py'`: ran 141 tests and passed.
- `python -m atlas3r train synthetic-overfit --output build/smoke/phase4b_synthetic_overfit --steps 1 --batch-size 1 --num-samples 2 --width 16 --height 12 --device cpu --log-every 1`: passed and wrote `checkpoint_last.pt`.
- `python -m atlas3r smoke checkpoint-tsdf --checkpoint build/smoke/phase4b_synthetic_overfit/checkpoint_last.pt --output build/smoke/phase4b_checkpoint_tsdf --width 16 --height 12 --device cpu`: passed and wrote predicted/target TSDF plus preview artifacts.
- `python -m atlas3r inspect tsdf-output --input build/smoke/phase4b_checkpoint_tsdf --mode surface`: passed.
- `git diff --check`: passed; Git warned that changed files will be converted
  from LF to CRLF in the working tree.
- `Get-Command make`: `make` is not recognized on PATH, so `make test`,
  `make lint`, `make typecheck`, and `make smoke` were not run.

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
- Phase 4A: optional PyTorch synthetic-overfit training MVP with checkpoint,
  metrics, prediction sample, and dependency-free preview.
- Phase 4B: trained-checkpoint inference bridge into `DepthObservation` and
  CPU TSDF smoke comparison artifacts.

## Current Known Gaps

- No final SMGT transformer, real-world inference bridge, real datasets,
  teacher model downloads, external model integration, or coherent sequence smoke.
- The tiny trained model remains synthetic-only and not usable for realtime
  mapping or real captures.
- No video decoding, live camera runtime, runtime scheduler changes, TSDF
  internals changes, GLB/PLY export, object-aware fusion, web server, notebook,
  or broad inspection bundle was added.
