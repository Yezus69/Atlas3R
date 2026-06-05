# Phase 6C True Persistent Incremental TSDF Backend Report

Branch: `codex/phase6c-true-incremental-tsdf-backend`

Base branch: `codex/phase6b-real-capture-incremental-mapper`

## Result

Phase 6C replaces the default incremental rebuild path with a persistent CPU
TSDF backend:

```text
measured atlas3r_recording
  -> selected DepthObservation records
  -> fixed offline grid bounds for this diagnostic run
  -> one persistent dense CPU TSDF state
  -> one-observation map updates
  -> final TSDF artifacts + batch CPU comparison
```

No model, teacher wrapper, fake depth, fake pose, live camera API, hidden
surface completion, object fusion, generated data artifact, or realtime/mm
accuracy claim was added.

## Code Changed

- Added `src/atlas3r/mapping/incremental_tsdf.py` with
  `IncrementalTSDFConfig`, `IncrementalTSDFUpdateStats`, and
  `PersistentIncrementalTSDFMapper`.
- The mapper allocates `tsdf_flat`/`weight_flat` once, precomputes
  `centers_world_m` once, reuses `integrate_depth_observation(...)`, tracks
  unique source frame IDs, and reports deterministic array-byte counters.
- Extended `runtime fuse-recording` with
  `--backend cpu-persistent|cpu-rebuild`; incremental mode defaults to
  `cpu-persistent`, while batch mode rejects backend selection.
- Split incremental runtime code into backend-specific modules so no new
  monolithic source file was introduced.
- `cpu-persistent` writes Phase 6B artifact paths plus
  `backend_comparison.json`; per-frame events include backend,
  implementation, observation load/update latency, observation count, observed
  voxel count, and `surface_point_count=null` because surface extraction is
  intentionally final-only.
- Added focused mapper, runtime, CLI, comparison, and status-handoff tests.
- Updated API contracts, decisions, progress, active task, and next-task docs.

## Real Run Evidence

Command run:

```bash
python -m atlas3r runtime fuse-recording \
  --recording runs/phase6a_recording_freiburg1_xyz_val/recording \
  --output runs/phase6c_persistent_fuse_recording_freiburg1_xyz_val \
  --pose-source recording \
  --depth-source recording \
  --max-frames 120 \
  --keyframe-stride 1 \
  --voxel-size-m 0.05 \
  --truncation-voxels 3.0 \
  --export-point-cloud \
  --export-mesh auto \
  --mode incremental \
  --backend cpu-persistent
```

Result:

- Source: existing Phase 6A TUM RGB-D `freiburg1_xyz` validation recording.
- Frames fused: 120, frame IDs 676 through 795.
- Output: `runs/phase6c_persistent_fuse_recording_freiburg1_xyz_val`.
- Surface points: 3,136; `surface_points.ply` written under ignored `runs/`.
- Mesh: not exported because optional `scikit-image` is unavailable;
  `mesh_status.json` contains the install hint.
- Persistent map update latency, milliseconds:
  - p50/p95/max: `28.83565` / `31.63879` / `35.3636`.
  - mean: `29.02168916666667`.
- Observation load latency p50/p95/max:
  `15.2357` / `16.67137` / `82.8656` ms.
- Offline setup/export/report costs:
  - fixed grid bounds precompute: `273.1778` ms.
  - persistent TSDF initialization: `5.4839` ms.
  - final surface extraction: `3.2966` ms.
  - backend comparison: `3407.139` ms.
  - geometry export/report path: `262.5581` ms.
  - total pipeline: `9357.1067` ms.
- Deterministic array-byte counters:
  - selected observations: `27,648,000`.
  - persistent float64 TSDF state: `4,914,432`.
  - precomputed voxel centers: `7,371,648`.
  - saved float32 TSDF arrays: `2,457,216`.
  - surface arrays: `62,720`.

## Phase 6B Comparison

Phase 6B rebuild timing on the same recording was:

- observation load p50/p95/max: `14.84` / `16.82713` / `51.109` ms.
- map update p50/p95/max: `1427.58275` / `3333.022785` / `3499.7332` ms.
- total pipeline: `195992.3747` ms.

Phase 6C isolated map update p95 is `31.63879` ms versus Phase 6B
`3333.022785` ms. That is the expected improvement from removing
rebuild-per-keyframe work. This is still not a live mapper claim: fixed bounds,
recording preloads, final surface extraction, batch comparison, and export are
outside the per-frame update loop, and max map update exceeded 33 ms.

## Backend Comparison

`backend_comparison.json` compared `cpu-persistent` final volume against batch
CPU TSDF on the same selected observations and fixed grid.

- Same grid shape: `true`, grid shape `[81, 79, 48]`.
- Common observed voxels: `11096`.
- Max/mean absolute TSDF delta on common observed voxels: `0.0` / `0.0`.
- Max/mean absolute weight delta: `0.0` / `0.0`.
- Observed voxel count delta: `0`.
- Surface point count delta: `0`.
- Point-cloud count delta: `0`.
- `matches_batch_within_tolerance=true`.
- Truth flags: `accuracy_report=false`, `performance_report=false`,
  `realtime_claim=false`.

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

- Format: passed; 170 files left unchanged.
- Format check: passed; 170 files already formatted.
- Ruff check: passed.
- Mypy: passed with no issues in 128 source files.
- Unit discovery: passed 218 tests. The pre-existing optional
  Torch/einops import warning appeared.
- `git diff --check`: passed; Git warned changed LF files will convert to CRLF.
- No `make` executable was available in this Windows shell, so `make test`,
  `make lint`, and `make typecheck` were not run.

## Blockers Before Apartment Live Mapping

- Bounds are still fixed offline from selected observations; live mapping needs
  bounded allocation or safe volume shifting.
- The backend is dense NumPy CPU TSDF, not a GPU/Metal/CUDA mapper.
- There is no live camera scheduler, queue/backpressure policy, or sensor API.
- Full observed-surface extraction and mesh export are final/report costs, not
  live incremental mesh chunks.
- No object-aware fusion, glTF/game-engine export, loop closure, or benchmark
  accuracy report exists.
