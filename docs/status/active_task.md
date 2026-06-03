# Active task

Goal: Phase 5B teacher-signal forge plus mapping bridge.

Preflight audit:

- `git status --short`: clean.
- `git branch --show-current`: `codex/phase5b-teacher-signal-forge-mapping-bridge`.
- `git diff --stat codex/phase4d-tum-eval-v2-training...HEAD`: Phase 5A diff
  spans clip-cache forge, temporal training, tests, and status docs.
- Source line audit over `src/**/*.py` found existing files above 450 lines:
  `api/contracts.py` 453, `data/synthetic_cube_room.py` 553,
  `data/tum_rgbd.py` 522, mapping sidecar/inspection helpers 459-498,
  `runtime/_fixture_inspection_helpers.py` 462,
  `training/checkpoint_inference.py` 469, and
  `training/tum_rgbd_temporal_train.py` 470.

Checklist:

- [x] Read Phase 5B goal, compact status docs, API contracts, and Phase 5A
  clip-cache/temporal context.
- [x] Create/use non-main branch
  `codex/phase5b-teacher-signal-forge-mapping-bridge`.
- [x] Add `atlas3r.teachers` signal-cache contracts and validation.
- [x] Add measured TUM clip-cache forge and local-folder ingest commands.
- [x] Add teacher-vs-clip inspection diagnostics.
- [x] Add teacher-signal to CPU TSDF mapping diagnostic.
- [x] Add focused fixture tests and CLI help coverage.
- [x] Run required format/lint/type/unit/diff checks plus make checks if
  available.
- [x] Attempt real Phase 5A clip-cache teacher forge/inspect/map runs if local
  caches exist, keeping outputs ignored.
- [x] Update API contracts, progress, Phase 5B report, and Phase 5C next task.

Scope note: the over-450-line audit is pre-existing repository shape, not
generated artifacts. Phase 5B will avoid adding monolithic files and will split
new implementation files before they approach the threshold.

Result: Phase 5B code, tests, real-data diagnostic attempts, compact report,
and Phase 5C handoff are complete. `make` targets were unavailable because
`make` is not installed in this Windows shell.
