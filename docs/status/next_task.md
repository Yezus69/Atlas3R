Phase 6C - Accelerated Incremental Mapper Prototype

Goal: create a bounded incremental mapper prototype that replaces the
diagnostic CPU TSDF rebuild-per-keyframe path for measured RGB-D/pose
observations.

Start from branch `codex/phase6b-real-capture-incremental-mapper`. Reload
`README.md`, `PLANS.md`, `docs/08_API_CONTRACTS.md`,
`docs/status/progress.md`, `docs/status/decisions.md`,
`docs/status/active_task.md`, and
`docs/status/phase6b_real_capture_incremental_mapper_report.md`.

Required work:

- Keep the measured-input boundary: no RGB-only fake depth, fake pose, hidden
  surface completion, realtime claim, or millimeter accuracy claim.
- Add a mapper interface that can support true incremental updates behind the
  current `DepthObservation` contract.
- Prototype either a vectorized CPU block update, GPU/Metal/CUDA TSDF path, or
  a small bounded-volume mapper; isolate optional acceleration dependencies.
- Preserve `runtime fuse-recording --mode batch|incremental` and add a new mode
  or backend selector only if the report names the backend honestly.
- Emit the same artifacts as Phase 6B: TSDF outputs, `surface_points.ply`,
  mesh status/optional mesh, `per_frame_events.jsonl`, latency, memory,
  quality, summary, and preview reports.
- Compare against the Phase 6B TUM evidence path and report p50/p95/max update
  latency plus deterministic array-byte counters.
- Add focused tests for backend selection, bounded state, report truth flags,
  and fallback behavior when optional acceleration is unavailable.
- Run the required format, lint, typecheck, unit, diff, and available `make`
  verification commands.

Expected output:

- A measured-recording incremental mapper backend with honest timing evidence.
- A compact report stating whether acceleration reduced the Phase 6B CPU TSDF
  rebuild bottleneck and what remains before live apartment mapping.
