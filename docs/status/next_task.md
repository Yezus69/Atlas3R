Phase 6D - Live-Ready Incremental Mapper Scheduler

Goal: turn the Phase 6C persistent CPU TSDF diagnostic into a live-ready
runtime slice with explicit scheduling, bounded memory, and honest end-to-end
latency accounting.

Start from branch `codex/phase6c-true-incremental-tsdf-backend`. Reload
`README.md`, `PLANS.md`, `docs/08_API_CONTRACTS.md`,
`docs/status/progress.md`, `docs/status/decisions.md`,
`docs/status/active_task.md`, and
`docs/status/phase6c_true_incremental_tsdf_backend_report.md`.

Required work:

- Keep the measured-input boundary: no fake RGB-only depth, fake pose, hidden
  surface completion, object fusion, live camera API, or realtime/mm claim.
- Add a runtime scheduler diagnostic that separates frame load, keyframe
  selection, map update, surface extraction, export, and report costs.
- Make queue/backpressure and dropped-keyframe behavior explicit with bounded
  counters in `runtime_events.jsonl`.
- Preserve `runtime fuse-recording --mode incremental --backend
  cpu-persistent|cpu-rebuild` for regression and comparison.
- Add tests for scheduler counters, bounded queues, no per-frame surface
  extraction in the fast path, and truth flags.
- Run the required format, lint, typecheck, unit, diff, and available `make`
  verification commands.

Expected output:

- A measured-recording scheduler diagnostic that states whether the separated
  map update stays near 30 FPS and what still blocks live apartment mapping.
