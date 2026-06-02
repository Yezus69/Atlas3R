# Progress Summary

This file is a rolling current-state summary, not an append-only transcript.
Detailed history belongs in git commits, tests, and older revisions.

## Current State

- Current phase: after Phase 3B.
- Latest completed implementation: dependency-free RGB frame-source boundary
  with NPZ and binary PPM sequence loaders emitting `FramePacket` records.
- Next task pointer: Phase 3C - FramePacket Clip Builder for Student Boundary.
- Architecture focus remains:
  `RGB FramePacket -> teacher/student geometry prediction -> DepthObservation /
  FramePrediction -> runtime scheduler -> TSDF/surfel/object map -> mesh/world output`.

## Latest Verified Test State

Most recent pre-cleanup full run visible in the old log:

- `python -m ruff format --check src tests`: passed.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 52 source files.
- `python -m unittest discover -s tests -p 'test_*.py'`: ran 107 tests and passed.
- `git diff --check`: passed; Git warned that changed files will be converted
  from LF to CRLF in the working tree.
- `make` was unavailable on PATH. `Get-Command make` reported the term was not
  recognized; `where.exe make` reported no files for the pattern.

Latest Phase 3B.1 cleanup verification in this working tree:

- `python -m ruff format --check src tests`: passed with 74 files already formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 52 source files.
- `python -m unittest discover -s tests -p 'test_*.py'`: ran 109 tests and passed.
- `Get-Command make`: `The term 'make' is not recognized as the name of a
  cmdlet, function, script file, or operable program.`
- `where.exe make`: `INFO: Could not find files for the given pattern(s).`
- `make test`, `make lint`, and `make typecheck` were not run because `make`
  is not available on PATH.
- `git diff --check`: passed; Git warned that changed files will be converted
  from LF to CRLF in the working tree.

## Compact Phase Ledger

- Skeleton: package layout, CLI entry point, Makefile targets, initial tests.
- Phase 0A: NumPy contracts, validation helpers, coordinate math, DenseMatchSet.
- Phase 0B: deterministic synthetic cube-room fixture and smoke session writer.
- Phase 0C: dependency-free `.atlas3r` session reader and HTML/SVG preview.
- Phase 0D: pure-NumPy CPU TSDF reference smoke artifacts.
- Phase 0E: dependency-safe geometry teacher adapter contracts and stubs.
- Phase 1A: summaries-first TeacherPrediction cache and adapter runner skeleton.
- Phase 1B: `fixture-cube-room` adapter producing deterministic synthetic caches.
- Phase 1C: opt-in teacher cache array payloads and cache inspection.
- Phase 1D: full-array teacher cache replay into CPU TSDF.
- Phase 1E: observed-only TSDF MeshChunk sidecar.
- Phase 2A: observed-only WorldMap sidecar wrapping the MeshChunk sidecar.
- Phase 2B: complete CPU TSDF output folder inspection.
- Phase 2C: public `DepthObservation` mapper input contract.
- Phase 2C.1: mapper boundary cleanup with synthetic converter and TSDF grid helpers.
- Phase 2D: deterministic single-threaded runtime fixture scheduler smoke.
- Phase 2E: runtime fixture output inspection.
- Phase 3A: NumPy-only student clip input/output boundary and shape-only stub.
- Phase 3B: dependency-free RGB frame-source boundary and NPZ/PPM smoke path.
- Phase 3B.1: context-budget cleanup for status/API docs and guard tests.

## Current Known Gaps

- No real neural inference, model training, datasets, weights, downloads, or
  vendored third-party model code.
- External teacher adapters remain dependency-safe stubs except the synthetic
  `fixture-cube-room` adapter.
- CPU TSDF outputs, MeshChunk sidecars, WorldMap sidecars, and inspections are
  deterministic diagnostic artifacts, not accuracy or performance reports.
- No GLB/PLY export, marching cubes, object-aware fusion, GPU/CUDA/Metal mapper,
  web server, notebook, video decoding, or live camera runtime.
- Runtime fixture timing and memory counters are deterministic placeholders, not
  process profiling or real-time throughput measurements.
- The shape-only student stub is not learned inference and is not usable for
  mapping.
- RGB frame-source smoke fixtures validate ingestion plumbing only and do not
  call the student model, runtime scheduler, or mapper.

## Phase 3B.1 Cleanup Update

Changed files:

- `AGENTS.md`
- `docs/08_API_CONTRACTS.md`
- `docs/status/active_task.md`
- `docs/status/progress.md`
- `docs/status/decisions.md`
- `docs/status/next_task.md`
- `tests/unit/test_repo_context_budget.py`

Commands run:

- `python -m ruff format --check src tests`
- `python -m ruff check tests/unit/test_repo_context_budget.py --fix`
- `python -m ruff check src tests`
- `python -m mypy src`
- `python -m unittest discover -s tests -p 'test_*.py'`
- `Get-Command make`
- `where.exe make`
- `git diff --check`

Results:

- Markdown budgets are within limits: AGENTS 123/140, PLANS 150/220, API
  contracts 450/450, active task 18/80, progress 120/180, decisions 36/180,
  next task 94/150.
- Ruff format, Ruff lint, mypy, full unittest discovery, and `git diff --check`
  pass.
- New line-budget guard test is included in the 109-test discovery run.
- `make` commands are unavailable for the reasons recorded above.
- No public source API, CLI command, or runtime behavior was intentionally changed.

Known gaps:

- Cleanup only; no source API, CLI command, runtime behavior, model behavior, or
  test behavior is intentionally changed.
