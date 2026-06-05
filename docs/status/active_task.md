# Active Task - Phase 6D Sparse Block TSDF Live-Replay Diagnostic

Goal: add a sparse/chunked incremental TSDF backend that grows online from one
measured `DepthObservation` at a time, preserves dense `cpu-persistent` and
`cpu-rebuild` baselines, compares sparse output against dense persistent output
on the same small recording, and reports apartment-scale memory/timing
diagnostics without realtime, accuracy, mapping-ready, or hidden-geometry
claims.

Branch: `codex/phase6d-sparse-block-tsdf-live-replay`

Checklist:

- [x] Start from `codex/phase6c-true-incremental-tsdf-backend` and create the
  Phase 6D branch.
- [x] Read the Phase 6D goal file and constrained repo context.
- [x] Add a mapping-only sparse block TSDF backend behind `DepthObservation`.
- [x] Extend `runtime fuse-recording --mode incremental --backend cpu-sparse`
  while preserving `cpu-persistent` and `cpu-rebuild`.
- [x] Write sparse backend artifacts, per-frame events, memory counters, and
  final observed surface output.
- [x] Add sparse-vs-dense persistent diagnostic comparison on the same selected
  observations.
- [x] Add apartment-scale sparse TSDF stress diagnostic command/output.
- [x] Add focused mapper, runtime, comparison, stress, truth-flag, and CLI tests.
- [x] Run required format, lint, typecheck, unit, diff, and available make
  verification commands.
- [x] Run the existing Phase 6A TUM recording evidence path if present and
  record results or the exact missing path.
- [x] Update compact progress, decisions, API contracts, Phase 6D report, and
  next-task handoff.
