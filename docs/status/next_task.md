Phase 6B - Live/Video Input and Calibration Capture

Goal: add a practical live/video input boundary and apartment capture path that
produces validated `atlas3r_recording` folders for Phase 6A fusion. This is not
another training run or teacher-wrapper task.

Start from branch `codex/phase6a-product-slice-mapper-recording-mesh`. Reload
`README.md`, `PLANS.md`, `docs/08_API_CONTRACTS.md`,
`docs/status/progress.md`, `docs/status/decisions.md`,
`docs/status/active_task.md`, and
`docs/status/phase6a_product_slice_mapper_recording_mesh_report.md`.

Required work:

- Add a dependency-safe video/local-camera ingestion boundary that can emit
  RGB frames plus calibration metadata into the recording schema.
- Add a clear calibration capture path for intrinsics and metric scale; do not
  claim metric accuracy without a named calibration/evaluation report.
- Support an apartment capture workflow that writes `atlas3r_recording.json`
  and `frames.jsonl`, then calls `atlas3r recording validate`.
- Keep optional camera/video dependencies isolated with clear install errors.
- Add a smoke fixture for video-like input and tests for calibration metadata,
  path safety, timestamp ordering, and failure modes.
- Run `runtime fuse-recording` only when measured depth+pose or an explicitly
  configured sensor source is available; do not fake depth, pose, or hidden
  geometry.

Expected output:

- Validated recording folders from live/video-like inputs.
- A short capture report with commands, calibration source, missing sensors, and
  exact blockers for measured depth/pose if unavailable.
- Updated compact status docs and API contracts.
