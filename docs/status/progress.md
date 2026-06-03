# Progress Summary

This file is a rolling current-state summary, not an append-only transcript.
Detailed history belongs in git commits, tests, and older revisions.

## Current State

- Current phase: Phase 5C external teacher runner bootstrap complete; Phase 5D
  teacher-signal training is next.
- Latest implementation: dependency-isolated external teacher runner contracts,
  shared external signal-cache writer, Depth Pro runner with explicit external
  checkpoint configuration, VGGT local-output ingestion, CLI commands, API docs,
  and fixture tests.
- Public teacher commands now include `forge-measured-tum`, `ingest-local`,
  `run-depth-pro`, `ingest-vggt-local`, `inspect-signals`, and `map-signals`.
- Teacher-signal caches remain the stable boundary. External outputs are
  pseudo-labels with `measured_geometry=false`, confidence/uncertainty fields,
  source clip IDs, frame IDs, timestamps, and Phase 5B payload validation.

## Latest Verified Test State

- `python -m ruff format src tests`: passed with 128 files unchanged.
- `python -m ruff format --check src tests`: passed with 128 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 93 source files.
- `python -m unittest discover -s tests -p "test_*.py"`: ran 175 tests and
  passed. A pre-existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git warned changed LF files will convert to CRLF.
- `make` commands were not run because `where.exe make` found no `make` in this
  Windows shell.

## Real-Data Evidence

- Phase 5A/5B TUM clip caches and measured teacher-signal diagnostics still
  exist locally under ignored `data/` and `runs/` paths.
- Phase 5C did not commit generated caches, model repos, checkpoints, or
  previews.
- `depth_pro` imports from an external local repo, but no external checkpoint URI
  is configured through `--checkpoint-uri` or `ATLAS3R_DEPTH_PRO_CHECKPOINT`.
  The real Depth Pro TUM run was intentionally skipped to avoid implicit weight
  loading. Fixture Depth Pro and VGGT outputs validate through inspect/map smoke.

## Compact Phase Ledger

- Skeleton through Phase 4: package skeleton, contracts, synthetic correctness,
  CPU TSDF diagnostics, dependency-safe adapter stubs, NumPy student boundary,
  optional Torch training MVP, checkpoint TSDF bridge, and TUM RGB-D debug
  train/eval.
- Phase 5A: canonical multi-view TUM clip cache plus tiny temporal center-depth
  and relative-translation training path.
- Phase 5B: stable teacher-signal cache plus measured forge, local ingest,
  source-clip inspection, and CPU TSDF map diagnostic bridge.
- Phase 5B.1: JSON-only inspection, explicit raw NPZ filename mapping,
  cache-local manifest resolution, invalid-manifest rejection, and deduplicated
  overlapping `map-signals` frame replay.
- Phase 5C: dependency-safe external teacher runner boundary, Depth Pro
  bootstrap, VGGT local ingest scaffold, validated external signal caches, and
  CLI/tests/docs handoff to Phase 5D.

## Current Known Gaps

- No real Depth Pro or VGGT output has been run in this branch because external
  checkpoint/local-output paths were not configured.
- No final SMGT transformer, teacher-signal training mix, video decoding, live
  camera runtime, object-aware mapping, GLB/PLY export, or benchmark
  accuracy/performance report.
