# Phase 6F Live Mesh Chunks Report

Branch: `codex/phase6f-live-mesh-chunks`

## Result

Phase 6F adds observed-only live triangle mesh chunk updates to measured replay:

```text
measured atlas3r_recording
  -> bounded live replay scheduler
  -> measured pose + measured depth DepthObservation
  -> sparse TSDF map
  -> dirty sparse blocks
  -> observed-only versioned mesh chunk NPZ/PLY payloads
```

The implementation does not claim RGB-only mapping readiness, realtime
readiness, millimeter accuracy, hidden/completed geometry, object-aware fusion,
loop closure, or neural student mapping.

## Artifacts

The measured evidence run wrote:

- `live_replay_events.jsonl`
- `live_replay_summary.json`
- `live_replay_report.md`
- `live_replay_latency_report.json`
- `sparse_tsdf/`
- `mesh_chunks/mesh_chunk_manifest.json`
- `mesh_chunks/mesh_chunk_updates.jsonl`
- `mesh_chunks/chunks/*.npz`
- `mesh_chunks/chunks/*.ply`
- `surface_points.ply`

Mesh chunks use stable `block_<x>_<y>_<z>` IDs, monotonic versions per chunk,
NPZ array payloads, optional PLY debug payloads, source frame IDs, voxel size,
coordinate frame, uncertainty/confidence summaries, and conservative truth
flags.

## Profiling

First Phase 6F baseline before optimization had map+mesh p95 `556.001 ms`.
The implemented optimization batches sparse active-voxel snapshot creation once
per mesh update instead of rebuilding mapper snapshots per dirty chunk. The
optimized baseline-compatible profile reduced map+mesh p95 to `222.709 ms`
(`60.0%` lower). The 33 ms preview target was not met.

| Profile | Frames | Keyframes | Map updates | Mesh chunk updates | Active blocks | Active voxels | Chunks | Vertices | Triangles | Map p50/p95/max ms | Mesh p50/p95/max ms | Total pipeline ms | Approx memory bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| baseline-compatible | 120 | 120 | 120 | 6044 | 142 | 10165 | 117 | 27944 | 13972 | 66.090 / 73.695 / 80.558 | 96.040 / 154.331 / 169.743 | 45363.277 | 52816568 |
| preview/live-friendlier | 120 | 60 | 60 | 2629 | 145 | 9118 | 118 | 26512 | 13256 | 31.505 / 35.041 / 36.392 | 76.750 / 136.029 / 147.004 | 18084.572 | 22851136 |
| coarse-live | 120 | 40 | 40 | 1502 | 117 | 7511 | 97 | 22304 | 11152 | 20.129 / 22.572 / 22.917 | 71.898 / 113.427 / 114.610 | 10062.025 | 13017880 |

The remaining bottleneck is mesh chunk update/export latency. Sparse candidate
generation also remains visible: optimized preview `sparse_candidate_voxel_coords`
p95 is still reflected inside map p95, which is slightly above 33 ms.

## Evidence Command

```bash
python -m atlas3r runtime live-replay-recording \
  --recording runs/phase6a_recording_freiburg1_xyz_val/recording \
  --output runs/phase6f_live_mesh_freiburg1_xyz_val \
  --target-fps 30 \
  --max-frames 120 \
  --mapper-backend cpu-sparse \
  --map-keyframe-stride 1 \
  --max-capture-queue 4 \
  --max-map-queue 2 \
  --drop-policy oldest \
  --voxel-size-m 0.05 \
  --truncation-voxels 3.0 \
  --pixel-stride 8 \
  --export-point-cloud \
  --export-mesh-chunks \
  --mesh-format ply
```

Result: `117` active chunks, `6044` chunk update events, `27944` vertices,
`13972` triangles, nonzero loadable NPZ chunks, nonzero PLY chunks, measured
depth/pose used, and observed-only truth flags.

## Verification

Focused commands passed before docs:

```bash
python -m ruff format src tests
python -m ruff check src tests
python -m mypy src
python -m unittest tests.unit.test_mesh_chunks tests.unit.test_sparse_tsdf_meshing tests.unit.test_live_replay_mesh_chunks tests.unit.test_sparse_tsdf_mapper tests.unit.test_live_replay_scheduler tests.unit.test_cli
```

Full final verification is recorded in `docs/status/progress.md`.
