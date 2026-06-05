# Active Task - Phase 6C True Persistent Incremental TSDF Backend

Goal: replace the Phase 6B rebuild-per-keyframe incremental path with a true
persistent CPU TSDF backend that integrates only each new measured
`DepthObservation`, compares final output against batch CPU TSDF, and reports
honest timing without realtime, accuracy, or hidden-geometry claims.

Branch: `codex/phase6c-true-incremental-tsdf-backend`

Checklist:

- [x] Start from `codex/phase6b-real-capture-incremental-mapper` and create the
  Phase 6C branch.
- [x] Read the Phase 6C goal file and constrained repo context.
- [x] Add a mapping-only persistent incremental TSDF backend behind the
  existing `DepthObservation` contract.
- [x] Extend `runtime fuse-recording --mode incremental` with
  `--backend cpu-persistent|cpu-rebuild`, defaulting incremental to
  `cpu-persistent` while preserving Phase 6B `cpu-rebuild`.
- [x] Emit Phase 6B artifact paths plus `backend_comparison.json` for
  persistent-vs-batch CPU TSDF deltas.
- [x] Keep per-frame events focused on observation load and one-observation map
  update latency; avoid per-frame surface extraction on the persistent fast path.
- [x] Add focused mapper, runtime artifact, comparison, truth-flag, and CLI
  validation tests.
- [x] Run required format, lint, typecheck, unit, diff, and available make
  verification commands.
- [x] Run the Phase 6A TUM recording evidence path if it exists and record real
  p50/p95/max map update timing.
- [x] Update compact progress, decisions, API contracts, Phase 6C report, and
  next-task handoff.
