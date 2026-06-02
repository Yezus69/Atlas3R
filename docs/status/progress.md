# Progress Summary

This file is a rolling current-state summary, not an append-only transcript.
Detailed history belongs in git commits, tests, and older revisions.

## Current State

- Current phase: Phase 3C complete; next pointer is Phase 3D.
- Latest implementation: dependency-free `FramePacket` -> `StudentClipInput`
  bridge in `atlas3r.data.student_clip_from_frame_packets`.
- The bridge preserves input order, rejects empty/non-packet/duplicate/mismatched
  clips, stacks `rgb_model` as `1,T,3,H,W`, stacks `K_model` as `1,T,3,3`, and
  emits compact deterministic metadata.
- Shape-only student forward accepts bridge clips and remains a truth-boundary
  stub with `learned_inference=false` and `usable_for_mapping=false`.
- Architecture focus remains:
  `RGB FramePacket -> teacher/student geometry prediction -> DepthObservation /
  FramePrediction -> runtime scheduler -> TSDF/surfel/object map -> mesh/world output`.

## Latest Verified Test State

Phase 3C verification in this working tree:

- `python -m ruff format --check src tests`: passed with 76 files already formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 53 source files.
- `python -m unittest discover -s tests -p 'test_*.py'`: ran 118 tests and passed.
- `git diff --check`: passed; Git warned that changed files will be converted
  from LF to CRLF in the working tree.
- `Get-Command make`: `The term 'make' is not recognized as the name of a
  cmdlet, function, script file, or operable program.`
- `where.exe make`: `INFO: Could not find files for the given pattern(s).`
- `make test`, `make lint`, and `make typecheck` were not run because `make`
  is not available on PATH.

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
- Phase 3C: minimal FramePacket clip builder for the student boundary.

## Current Known Gaps

- No real neural inference, model training, datasets, weights, downloads, or
  vendored third-party model code.
- External teacher adapters remain dependency-safe stubs except the synthetic
  `fixture-cube-room` adapter.
- The student clip bridge is shape/contract plumbing only; it does not feed
  mapper, runtime, TSDF, export, or inspection paths.
- CPU TSDF outputs, MeshChunk sidecars, WorldMap sidecars, and inspections are
  deterministic diagnostic artifacts, not accuracy or performance reports.
- No GLB/PLY export, marching cubes, object-aware fusion, GPU/CUDA/Metal mapper,
  web server, notebook, video decoding, or live camera runtime.
- Runtime fixture timing and memory counters are deterministic placeholders, not
  process profiling or real-time throughput measurements.
- The shape-only student stub is not learned inference and is not usable for
  mapping.

## Phase 3C Update

Changed files:

- `src/atlas3r/data/student_clip.py`
- `src/atlas3r/data/__init__.py`
- `tests/unit/test_student_clip.py`
- `tests/unit/test_repo_context_budget.py`
- `tests/synthetic/test_session_inspect.py`
- `docs/08_API_CONTRACTS.md`
- `docs/status/active_task.md`
- `docs/status/progress.md`
- `docs/status/decisions.md`
- `docs/status/next_task.md`

Results:

- Added public, lazy-exported `student_clip_from_frame_packets`.
- Added focused bridge tests for success, ordering, duplicate IDs, empty clips,
  non-packets, mismatched shapes, no input-array mutation, shape-only forward,
  and dependency-safe imports.
- Updated API contracts, ADR index, and the next-task prompt for Phase 3D.
- No heavy dependency, CLI, mapper, runtime, TSDF, export, or inspection-bundle
  changes were introduced.
