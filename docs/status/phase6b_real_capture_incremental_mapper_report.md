# Phase 6B Real Capture Incremental Mapper Report

Branch: `codex/phase6b-real-capture-incremental-mapper`

Base branch: `codex/phase6a-product-slice-mapper-recording-mesh`

## Result

Phase 6B added the real capture boundary and incremental timing path:

```text
sensor_capture/
  -> atlas3r recording from-sensor-folder
  -> atlas3r recording validate
  -> atlas3r runtime fuse-recording --mode incremental
  -> TSDF artifacts, surface_points.ply, mesh status/optional OBJ,
     per_frame_events.jsonl, latency/memory/quality/summary reports
```

No model training, teacher wrapper, fake depth, fake pose, hidden-geometry
completion, generated capture, checkpoint, image, video, PLY, OBJ, or run
artifact was committed.

## Code Changed

- Added `atlas3r recording from-sensor-folder` for dependency-light measured
  capture folders with `sensor_capture.json`, `frames.jsonl`, safe relative
  RGB/depth paths, calibrated K, optional measured depth, and optional measured
  `T_world_camera`.
- Added `src/atlas3r/recording/sensor_folder.py` with input validation for
  coordinate convention, calibration/truth metadata, image dimensions, finite
  timestamps/K/poses, and positive PNG depth scale.
- Added `runtime fuse-recording --mode batch|incremental`; batch remains the
  default, while incremental loads selected keyframes one at a time and records
  per-frame observation load and map update timings.
- Added focused tests for sensor-folder import, unsafe paths, missing
  depth/pose, validation after import, incremental reports/artifacts, batch
  mode, optional mesh fallback/export behavior, and CLI help.
- Updated API contracts, progress, decisions, active task, report, and next
  task handoff.

## Real Run Evidence

Command run:

```bash
python -m atlas3r runtime fuse-recording \
  --recording runs/phase6a_recording_freiburg1_xyz_val/recording \
  --output runs/phase6b_incremental_fuse_recording_freiburg1_xyz_val \
  --pose-source recording \
  --depth-source recording \
  --max-frames 120 \
  --keyframe-stride 1 \
  --voxel-size-m 0.05 \
  --truncation-voxels 3.0 \
  --export-point-cloud \
  --export-mesh auto \
  --mode incremental
```

Result:

- Source: existing Phase 6A TUM RGB-D `freiburg1_xyz` validation recording.
- Frames fused: 120, frame IDs 676 through 795.
- `per_frame_events.jsonl`: 120 records.
- Surface points: 3,136; `surface_points.ply` written under ignored `runs/`.
- Mesh: not exported because optional `scikit-image` is unavailable;
  `mesh_status.json` contains the install hint.
- Valid measured depth pixels: 1,714,731; valid ratio `0.7442408854166667`.
- Incremental timing, wall-clock diagnostic milliseconds:
  - observation load p50/p95/max: `14.84` / `16.82713` / `51.109`.
  - map update p50/p95/max: `1427.58275` / `3333.022785` / `3499.7332`.
  - total pipeline: `195992.3747` ms.
- Deterministic array-byte counters:
  - observations: 27,648,000.
  - TSDF arrays: 2,457,216.
  - surface arrays: 62,720.

## Bottleneck

The bottleneck is clearly CPU TSDF map update, not capture ingestion. Phase 6B
incremental mode rebuilds the CPU TSDF from selected observations per frame, so
it is useful timing instrumentation but not a true realtime incremental mapper.
Phase 6C should focus on an accelerated or bounded incremental mapper backend.

## Verification

Commands run:

```bash
python -m ruff format src tests
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m unittest discover -s tests -p "test_*.py"
git diff --check
where.exe make
Get-Command make -ErrorAction SilentlyContinue
```

Results:

- Format, lint, mypy, full unit discovery, and `git diff --check` passed.
- Unit discovery passed 213 tests with the pre-existing optional Torch/einops
  import warning.
- No `make` executable was available in this Windows shell, so `make test`,
  `make lint`, and `make typecheck` were not run.
