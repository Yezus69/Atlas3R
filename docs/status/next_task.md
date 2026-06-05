Phase 6E - Live Capture Replay Scheduler And Camera Adapter Boundary

Goal: build the first live-oriented runtime scheduler around the measured
recording/sparse-map path, plus a dependency-safe actual camera adapter
boundary. This phase should make queueing, frame dropping, replay pacing, and
adapter installation errors explicit. It must not claim realtime mapping unless
a measured report proves it.

Start from branch `codex/phase6d-sparse-block-tsdf-live-replay`. Reload
`README.md`, `PLANS.md`, `docs/08_API_CONTRACTS.md`,
`docs/status/progress.md`, `docs/status/decisions.md`,
`docs/status/active_task.md`, and
`docs/status/phase6d_sparse_block_tsdf_live_replay_report.md`.

Required work:

- Preserve measured-input truth boundaries: no fake RGB-only depth, fake pose,
  hidden surface completion, object fusion, realtime/mm/accuracy claims, or
  generated data commits.
- Keep `runtime fuse-recording --mode incremental --backend
  cpu-persistent|cpu-rebuild|cpu-sparse` stable for regression.
- Add a replay scheduler diagnostic that can pace an `atlas3r_recording` at a
  requested frame rate, select mapping keyframes, bound queue depth, and record
  explicit dropped-frame/keyframe reasons.
- Keep map updates on the existing measured-depth path; use `cpu-sparse` as the
  live-oriented mapper unless a test requires a dense baseline comparison.
- Add a dependency-safe camera adapter interface under `src/atlas3r/runtime/`
  that can report unavailable optional capture dependencies with install hints
  and no import-time crash.
- Add tests for bounded queues, replay pacing metadata, drop counters, adapter
  unavailable status, truth flags, and unchanged existing backends.
- Run required format, lint, typecheck, unit, diff, and available `make`
  verification commands.

Expected output:

- A measured replay scheduler report with per-stage latency, queue/drop
  counters, selected keyframes, mapper state counters, and explicit statements
  about why the result is or is not live mapping ready.
