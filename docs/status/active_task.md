# Active Task - Phase 6B Real Capture Incremental Mapper

Goal: add a dependency-light real sensor-folder import boundary and expose
incremental measured recording fusion timing without claiming realtime,
accuracy, hidden geometry completion, or mesh quality beyond measured artifacts.

Branch: `codex/phase6b-real-capture-incremental-mapper`

Checklist:

- [x] Start from `codex/phase6a-product-slice-mapper-recording-mesh` and create
  the Phase 6B branch.
- [x] Read the Phase 6B goal file and constrained repo context.
- [x] Add `atlas3r recording from-sensor-folder` with strict input validation
  and conversion to the existing `atlas3r_recording` format.
- [x] Add deterministic tiny sensor-capture fixture support for tests only.
- [x] Add `runtime fuse-recording --mode batch|incremental`; keep batch
  behavior and report incremental per-frame load/update timings honestly.
- [x] Preserve optional mesh export behavior and clear missing-dependency
  status for `scikit-image`.
- [x] Add focused importer, validation, runtime, mesh fallback, and CLI tests.
- [x] Run required format, lint, typecheck, unit, diff, and available make
  verification commands.
- [x] Run Phase 6A local TUM recording incremental fusion if the recording
  exists; otherwise record the missing-data blocker.
- [x] Update compact progress, decisions, API contracts, Phase 6B report, and
  next-task handoff.
