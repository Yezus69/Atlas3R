# Phase 5E Streaming Student Map Runtime Report

Branch: `codex/phase5e-streaming-student-map-runtime`

Base commit before Phase 5E edits: `a1fafa2`. The implementation commit is the
branch tip containing this report.

## What Changed

- Added `atlas3r runtime stream-student-map`.
- Built a unique chronological clip-cache stream and deterministic padded windows.
- Loaded Phase 5D `TemporalMetricNetV1` checkpoints and emitted one
  `DepthObservation` per unique frame.
- Fused observations with existing CPU TSDF helpers and wrote TSDF sidecars,
  ASCII PLY point clouds, preview HTML, runtime events, quality reports, and
  latency reports.
- Implemented `oracle` and diagnostic-only `student-relative` pose modes.

All reports keep `diagnostic_only=true`, `accuracy_report=false`,
`performance_report=false`, `realtime_claim=false`, and `mapping_ready=false`.

## Real Local Run

Inputs existed and the required CUDA command completed:

```bash
python -m atlas3r runtime stream-student-map \
  --checkpoint runs/phase5d_teacher_temporal_v1_tum/checkpoint_best.pt \
  --clip-cache data/tum_rgbd/freiburg1_xyz_clip_cache_val \
  --teacher-cache data/tum_rgbd/freiburg1_xyz_measured_teacher_val \
  --output runs/phase5e_stream_student_map_val \
  --device cuda \
  --max-frames 60 \
  --window-size 5 \
  --pose-mode both \
  --voxel-size-m 0.05
```

Artifacts:

- `runs/phase5e_stream_student_map_val/oracle/`
- `runs/phase5e_stream_student_map_val/student-relative/`
- Each mode contains `summary.json`, `quality_report.json`,
  `latency_report.json`, `runtime_events.jsonl`, `observations_summary.jsonl`,
  `tsdf/`, `point_cloud.ply`, and `map_preview.html`.

## Diagnostic Metrics

Depth versus measured teacher cache, both pose modes:

- RMSE: `0.090826432 m`
- MAE: `0.050450069 m`
- AbsRel: `0.046490420`
- Overlap pixels: `858195`
- Within 1 mm / 5 mm / 1 cm / 5 cm / 10 cm: `1.793%`, `8.846%`,
  `17.536%`, `70.136%`, `89.566%`

CPU TSDF:

- Oracle: `8622` surface points, observed coverage `0.08431`
- Student-relative: `8732` surface points, observed coverage `0.08583`
- Measured teacher reference map: `3031` surface points

Latency diagnostics:

- Oracle model inference mean/p50/p95/max: `10.296 ms`, `4.147 ms`,
  `4.340 ms`, `374.137 ms`
- Student-relative model inference mean/p50/p95/max: `4.141 ms`, `4.094 ms`,
  `4.307 ms`, `6.162 ms`
- Total pipeline includes CPU TSDF, teacher-map comparison, export, and report
  writing; it is not a realtime measurement.

## Verification

- `python -m ruff format src tests`: passed, 142 files unchanged.
- `python -m ruff format --check src tests`: passed, 142 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed, 105 source files checked.
- `python -m unittest discover -s tests -p "test_*.py"`: passed, 186 tests;
  existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git emitted CRLF conversion warnings.
- `where.exe make`: no `make` found in this Windows shell, so `make test`,
  `make lint`, and `make typecheck` were not run.

## Remaining Blockers

- No realtime scheduler or bounded-memory GPU mapper is implemented.
- `student-relative` pose is a diagnostic translated pose around source anchors;
  it is not a validated tracker or mapping-ready pose estimate.
- Quality is only versus the provided measured teacher cache, not a benchmark
  accuracy report.
- PLY output is a point cloud from TSDF surface samples, not a triangle mesh.
