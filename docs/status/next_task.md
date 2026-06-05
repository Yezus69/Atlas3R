Phase 6F - Live Mesh Chunk Updates From Sparse Replay

Goal: build the next live-oriented product slice after Phase 6E by emitting
observed-only triangle mesh chunk update artifacts from the sparse TSDF replay
path, while preserving all measured-input truth boundaries.

Start from branch `codex/phase6e-live-replay-scheduler`. Reload `README.md`,
`PLANS.md`, `docs/08_API_CONTRACTS.md`, `docs/status/progress.md`,
`docs/status/decisions.md`, `docs/status/active_task.md`,
`docs/status/phase6e_live_replay_scheduler_report.md`, and
`docs/status/phase6e_repo_audit.md`.

Required work:

- Preserve `runtime fuse-recording --mode incremental --backend
  cpu-persistent|cpu-rebuild|cpu-sparse` and `runtime live-replay-recording`
  behavior.
- Add observed-only mesh chunk update artifacts for live replay, with stable
  chunk IDs, versions, source frame IDs, voxel size, coordinate frame,
  uncertainty percentiles, queue/update metadata, and no hidden-geometry or
  RGB-only mapping claim.
- Prefer a small dependency-safe meshing path or a documented optional
  dependency fallback; do not vendor third-party code or weights.
- Add tests for chunk schema, update ordering, no unbounded queue growth, truth
  flags, and existing runtime regression commands.
- Run format, lint, typecheck, unit tests, diff check, available make commands,
  and a tiny or real measured replay evidence run under ignored `runs/`.

Expected output:

- `live_replay_mesh_updates.jsonl` or equivalent observed-only chunk update log.
- Mesh chunk sidecars or chunk payloads loadable through existing API contracts.
- A concise Phase 6F report explaining measured inputs used, chunk/update
  counters, latency, bottlenecks, and why no realtime/mm/RGB-only accuracy claim
  is made.
