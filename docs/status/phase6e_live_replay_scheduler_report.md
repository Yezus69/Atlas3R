# Phase 6E Live Replay Scheduler Report

Branch: `codex/phase6e-live-replay-scheduler`

## Result

Phase 6E adds a live-oriented diagnostic scheduler:

```text
measured atlas3r_recording
  -> deterministic replay pacing
  -> bounded capture queue and pose updates
  -> bounded map queue and keyframe decisions
  -> measured DepthObservation only
  -> cpu-sparse TSDF update
  -> live replay events, summary, report, sparse artifacts
```

No RGB-only pose/depth, hidden geometry completion, object fusion, loop closure,
realtime claim, performance report, benchmark accuracy report, millimeter claim,
or generated data commit was added.

## Code Changed

- Added runtime capture adapter contracts and status reporting:
  `CaptureAdapterStatus`, `CaptureFrame`, `LiveCameraAdapter`,
  `OpenCVCameraAdapter`, and `ReplayRecordingAdapter`.
- `OpenCVCameraAdapter` does not import `cv2` at module import time and raises
  explicit dependency/config errors.
- Added `runtime capture-adapters list`.
- Added `runtime live-replay-recording` with target FPS metadata, simulated
  pacing by default, bounded capture/map queues, explicit drop reasons, explicit
  keyframe reasons, separate pose updates, sparse TSDF map updates, summary,
  event JSONL, report Markdown, sparse state artifacts, and optional PLY export.
- Existing `runtime fuse-recording` incremental backends remain unchanged and
  covered by regression tests.

## Real Run Evidence

Command run:

```bash
python -m atlas3r runtime live-replay-recording \
  --recording runs/phase6a_recording_freiburg1_xyz_val/recording \
  --output runs/phase6e_live_replay_freiburg1_xyz_val \
  --target-fps 30 \
  --max-frames 120 \
  --mapper-backend cpu-sparse \
  --map-keyframe-stride 1 \
  --max-capture-queue 4 \
  --max-map-queue 2 \
  --drop-policy oldest \
  --voxel-size-m 0.05 \
  --truncation-voxels 3.0 \
  --export-point-cloud
```

Result:

- Source: existing Phase 6A TUM RGB-D `freiburg1_xyz` validation recording.
- Frames seen/emitted: 120 / 120.
- Pose updates: 120 measured `T_world_camera`.
- Keyframes selected: 120.
- Map updates: 120 measured depth+pose `DepthObservation`s.
- Drops: 0 frames, 0 keyframes.
- Max capture/map queue depth observed: `1 / 1`.
- Sparse state: 142 active blocks, 10,165 active voxels, approximate state
  bytes `585,040`.
- Surface points: 2,950; `surface_points.ply` written under ignored `runs/`.
- Map update latency ms p50/p95/max: `63.936 / 68.0898 / 73.9971`.
- Observation load latency ms p50/p95/max: `15.44545 / 17.04904 / 51.3417`.
- Total diagnostic pipeline: `9,627.9508` ms.
- Mesh: not exported; Phase 6E writes observed surface points, not triangles.

## Verification

Commands run:

```bash
python -m ruff format src tests
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m unittest tests.unit.test_live_replay_scheduler
python -m unittest tests.unit.test_capture_adapters
python -m unittest tests.unit.test_recording_runtime
python -m unittest tests.unit.test_cli
python -m unittest discover -s tests -p "test_*.py"
git diff --check
```

Results:

- Format: passed; final run left 183 files unchanged.
- Format check: passed.
- Ruff check: passed.
- Mypy: passed with no issues in 138 source files.
- Focused tests: passed 19 tests.
- Unit discovery: passed 233 tests. The pre-existing optional Torch/einops
  import warning appeared.
- `git diff --check`: passed; Git warned changed LF files will convert to CRLF.
- `make` commands were not run because `make` is unavailable in this Windows
  shell.

## Remaining Gaps

- CPU sparse map update p95 is about 68.1 ms on the measured replay, so this is
  not realtime mapping-ready.
- No live triangle mesh chunk updates or game-engine streaming assets exist yet.
- No object-aware fusion, loop closure, accelerated mapper, real camera hardware
  CI path, RGB-only student mapping readiness, or benchmark accuracy report
  exists.
