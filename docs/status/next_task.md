Phase 6H - Distill RGB Teacher Geometry Into Student Training Data

Goal: turn Phase 6G RGB teacher outputs into compact, validated temporal
training data for the SMGT student path, without treating teacher pseudo labels
as measured geometry.

Start from branch `codex/phase6g-rgb-teacher-map`. Reload `README.md`,
`PLANS.md`, `docs/08_API_CONTRACTS.md`, `docs/09_EVALUATION.md`,
`docs/status/progress.md`, `docs/status/decisions.md`,
`docs/status/active_task.md`, and
`docs/status/phase6g_rgb_teacher_map_report.md`.

Context:

- Phase 6G proved real VGGT RGB teacher-pseudo depth/pose/intrinsics can drive
  existing sparse TSDF fusion and observed mesh chunk export.
- Phase 6G is offline and teacher-heavy. It is not the final RGB-only student
  mapper, not realtime, not loop-closed, and not a benchmark accuracy report.
- Measured recording depth/pose may be used for diagnostic eval only, not for
  training targets unless the cache explicitly marks them measured.

Required work:

- Define a compact pseudo recording or teacher-signal export from Phase 6G runs
  that preserves RGB frame IDs, intrinsics, pseudo depth/sigma/confidence,
  pseudo `T_world_camera`, teacher metadata, source run path, and truth flags.
- Add validation and inspection for this export, including safe relative paths,
  finite arrays, consistent shapes, and explicit pseudo/measured fields.
- Build a small temporal dataset bridge that can feed existing student training
  code from Phase 6G pseudo labels while weighting pseudo targets lower than
  measured targets.
- Add dependency-safe tests with a fixture teacher run and at least one real
  exported Phase 6G cache inspection.
- Update contracts, evaluation notes, status docs, and a Phase 6H report.

Acceptance:

- A fixture `runtime map-rgb-teacher --teacher fixture-vggt` run can be exported
  to a validated temporal training cache and loaded by the student training data
  bridge.
- The real Phase 6G Freiburg RGB teacher run can be inspected/exported without
  copying model weights, vendoring external repositories, or weakening truth
  flags.
- Verification commands include format, lint, typecheck, unit tests, diff check,
  and the export/inspect evidence command.
