# Active Task - Phase 6E Live Replay Scheduler And Camera Adapter Boundary

Goal: add a measured-recording live replay scheduler with bounded queues,
explicit keyframe/drop decisions, sparse TSDF map updates, conservative truth
flags, and a dependency-safe camera adapter boundary.

Branch: `codex/phase6e-live-replay-scheduler`

Checklist:

- [x] Confirm the working tree is clean and create the Phase 6E branch.
- [x] Read the Phase 6E goal and relevant architecture, contracts, evaluation,
  and compact status docs.
- [x] Audit tracked repo files for current architecture, generated artifacts,
  compact reports, and stale/duplicate candidates.
- [x] Add dependency-safe runtime capture adapter contracts and OpenCV status
  handling without import-time `cv2`.
- [x] Add a live-style replay scheduler over existing `atlas3r_recording`
  loading and measured `DepthObservation` mapper input.
- [x] Add `runtime live-replay-recording` and `runtime capture-adapters list`
  CLI coverage while preserving `runtime fuse-recording` backends.
- [x] Write live replay events, summary, report, sparse TSDF artifacts, and
  conservative truth-boundary metadata.
- [x] Add focused scheduler, adapter, runtime smoke, CLI, and regression tests.
- [x] Update concise README, PLANS, API contracts, architecture, evaluation,
  progress, decisions, next-task, and Phase 6E report docs.
- [x] Run required format, lint, typecheck, unit, diff, available make, and
  real/tiny replay evidence commands.
- [x] Commit the completed Phase 6E work.
