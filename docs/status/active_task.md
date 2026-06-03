# Active task

Goal: Phase 5A real multi-view TUM RGB-D clip forge plus tiny temporal geometry
training path.

Checklist:

- [x] Finish pending Phase 4D report and branch from a non-main Phase 4D
  closeout commit.
- [x] Read the Phase 5A goal, compact status docs, API contracts, and relevant
  TUM/CLI/training/test modules.
- [x] Add `atlas3r.forge` with canonical multi-view clip-cache manifest,
  payload writer/reader validation, pointmap, and normal helpers.
- [x] Add `atlas3r forge tum-rgbd-clips` with deterministic JSON output.
- [x] Add lazy Torch clip-cache dataset with center-frame targets and
  `relative_T_center_camera`.
- [x] Add `TinyTemporalMetricNetV0` plus masked temporal losses and diagnostic
  metrics.
- [x] Add `atlas3r train tum-rgbd-temporal` with run artifacts, checkpoints,
  preview, summary, and truth-boundary metadata.
- [x] Add focused unit/CLI/optional Torch smoke tests and update API/status docs.
- [x] Forge real train/val TUM clip caches and complete the temporal-v0 CUDA
  run without committing generated artifacts.
- [x] Write the Phase 5A report and replace `next_task.md` with Phase 5B.

Result: Phase 5A implementation, diagnostic real-data run, verification, and
Phase 5B handoff docs are complete on
`codex/phase5a-real-multiview-forge-temporal`. Code/docs-only commit is pending.
