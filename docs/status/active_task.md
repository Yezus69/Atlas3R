# Active Task

Goal: Phase 5F real Depth Pro teacher data, mixed teacher-signal training, and
streaming student map comparison.

Branch: `codex/phase5f-real-depthpro-mixed-training-runtime`

Checklist:

- [x] Start from `codex/phase5e-streaming-student-map-runtime` and create the
  Phase 5F branch.
- [x] Read the Phase 5F goal and only the requested docs/source/tests.
- [x] Preflight required local TUM and Phase 5D artifacts.
- [x] Harden `teachers run-depth-pro` for device selection, real model tensor
  device transfer, output resizing, and unique-frame prediction reuse.
- [x] Add focused tests for Depth Pro dedupe, resize, CLI `--device`, and missing
  optional dependency behavior.
- [x] Run required format, lint, typecheck, unit, diff, and CLI help checks.
- [x] Attempt real Depth Pro availability/install/checkpoint discovery without
  vendoring code or weights.
- [x] If available, generate real Depth Pro teacher-signal caches, inspect, and
  map validation pseudo-labels.
- [x] Apply quality gate before mixed measured+pseudo training.
- [x] If gate passes, train mixed temporal student.
- [x] Export student pseudo-label cache, inspect it, and run streaming map
  comparison.
- [x] Write Phase 5F evidence report and update compact status files.
- [ ] Commit code/docs only after tests pass.

Known constraints:

- Do not fake Depth Pro outputs or relabel measured TUM depth as external
  pseudo-labels.
- Generated datasets, runs, checkpoints, `.npz` payloads, PLY files, previews,
  and downloaded weights stay ignored and uncommitted.
- All outputs remain diagnostic unless an explicit evaluation report supports
  stronger claims.
