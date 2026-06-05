# Phase 6D Sparse Block TSDF Live-Replay Report

Branch: `codex/phase6d-sparse-block-tsdf-live-replay`

Base branch: `codex/phase6c-true-incremental-tsdf-backend`

## Result

Phase 6D adds a sparse block TSDF backend:

```text
measured atlas3r_recording
  -> one DepthObservation at a time
  -> sparse blocks allocated lazily from integer voxel/block coordinates
  -> final observed surface + sparse state artifacts
  -> non-exact diagnostic comparison against dense cpu-persistent
  -> apartment-scale sparse/dense memory stress diagnostic
```

No model, teacher wrapper, fake depth, fake pose, hidden surface completion,
object fusion, CUDA/Metal dependency, generated data commit, realtime claim,
mapping-ready claim, performance report, or accuracy report was added.

## Code Changed

- Added `SparseTSDFConfig`, `SparseTSDFUpdateStats`, and
  `SparseBlockTSDFMapper` under `src/atlas3r/mapping/`.
- Sparse integration samples measured depth pixels with deterministic stride,
  unprojects surface points, deduplicates candidate voxels per frame, projects
  candidate voxel centers back to the measured depth image, and fuses clamped
  TSDF values into lazily allocated blocks.
- Duplicate frame IDs are skipped deterministically; the first observation for a
  frame ID owns the update and source-frame record.
- Added `runtime fuse-recording --mode incremental --backend cpu-sparse` while
  preserving `cpu-persistent` and `cpu-rebuild`.
- `cpu-sparse` writes `sparse_tsdf/sparse_tsdf_state.npz`, surface arrays,
  `surface_points.ply` when requested, reports, per-frame active block/voxel
  counters, and final-only surface extraction.
- Added sparse-vs-dense persistent comparison with sampled nearest-neighbor
  surface distances and 1 cm / 5 cm / 10 cm precision/recall-like percentages.
- Added `runtime sparse-tsdf-stress` for apartment-scale dense-memory estimates
  and sparse active-state diagnostics.

## Real Run Evidence

Command run:

```bash
python -m atlas3r runtime fuse-recording \
  --recording runs/phase6a_recording_freiburg1_xyz_val/recording \
  --output runs/phase6d_sparse_fuse_recording_freiburg1_xyz_val \
  --pose-source recording \
  --depth-source recording \
  --max-frames 120 \
  --keyframe-stride 1 \
  --voxel-size-m 0.05 \
  --truncation-voxels 3.0 \
  --export-point-cloud \
  --export-mesh auto \
  --mode incremental \
  --backend cpu-sparse
```

Result:

- Source: existing Phase 6A TUM RGB-D `freiburg1_xyz` validation recording.
- Frames fused: 120, frame IDs 676 through 795.
- Output: `runs/phase6d_sparse_fuse_recording_freiburg1_xyz_val`.
- Fixed dense bounds precomputed for sparse update loop: `false`.
- Sparse state: 142 active blocks, 10,165 active voxels, 72,704 allocated block
  voxels, approximate state bytes `585,040`.
- Surface points: 2,950; `surface_points.ply` written under ignored `runs/`.
- Sparse update latency, milliseconds p50/p95/max:
  `66.108 / 70.6655 / 73.3619`.
- Observation load latency p50/p95/max:
  `15.51155 / 17.08248 / 50.0748` ms.
- Final surface extraction: `5.4718` ms.
- Sparse-vs-dense comparison: `4,107.1554` ms after sparse replay.
- Total diagnostic pipeline: `14,036.9036` ms.
- Mesh: not exported; sparse backend currently exports observed surface points,
  not triangles.

## Sparse vs Dense Diagnostic

`backend_comparison.json` compared the sparse final surface against a dense
`cpu-persistent` baseline run after sparse replay on the same observations.

- Dense surface points: 3,136.
- Sparse surface points: 2,950.
- Dense persistent state bytes: `12,286,080`.
- Sparse state bytes: `585,040`.
- Sparse/dense state-memory ratio: `0.047618117414179296`.
- Sampled sparse-to-dense mean/p95 distance: `0.018038865381584536` /
  `0.050000011920928955` m.
- Sampled dense-to-sparse mean/p95 distance: `0.021584947485562812` /
  `0.05000007152563005` m.
- Symmetric Chamfer-like mean: `0.019811906433573674` m.
- Precision-like within 1 cm / 5 cm / 10 cm:
  `64.453125 / 94.873046875 / 99.90234375` percent.
- Recall-like within 1 cm / 5 cm / 10 cm:
  `64.453125 / 90.966796875 / 98.388671875` percent.

This is a representation diagnostic, not exact TSDF equality and not an
accuracy report.

## Apartment Stress Diagnostic

Command run:

```bash
python -m atlas3r runtime sparse-tsdf-stress \
  --output runs/phase6d_sparse_tsdf_stress_apartment \
  --room-size-m 10,10,3 \
  --voxel-size-m 0.05
```

Result:

- Dense apartment box shape: `[200, 200, 60]`.
- Dense voxel count: `2,400,000`.
- Dense TSDF/weight estimate: `19,200,000` bytes.
- Dense precomputed centers estimate: `57,600,000` bytes.
- Dense persistent estimated bytes: `76,800,000`.
- Sparse stress state: 146 active blocks, 36,653 active voxels, approximate
  state bytes `601,520`.
- Sparse/dense state-memory ratio: `0.007832291666666666`.
- Surface points: 12,240.

## Verification

Commands run:

```bash
python -m ruff format src tests
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m unittest tests.unit.test_sparse_tsdf_mapper tests.unit.test_recording_runtime tests.unit.test_cli
python -m unittest discover -s tests -p "test_*.py"
git diff --check
```

Results:

- Format: passed; final run left 177 files unchanged.
- Format check: passed; 177 files already formatted.
- Ruff check: passed.
- Mypy: passed with no issues in 134 source files.
- Focused tests: passed 12 tests.
- Unit discovery: passed 223 tests. The pre-existing optional Torch/einops
  import warning appeared.
- `git diff --check`: passed; Git warned changed LF files will convert to CRLF.
- `make test`, `make lint`, and `make typecheck` were not run because `make` is
  not available in this Windows shell.

## Still Blocks Live Apartment Mapping

- Sparse CPU updates are not realtime on the measured 120-frame run; p95 update
  latency is about 70.7 ms with deterministic stride 8.
- Sparse output is observed surface points plus sparse state, not live triangle
  mesh chunks.
- No live camera adapter, queue/backpressure scheduler, GPU/Metal/CUDA mapper,
  object-aware fusion, loop closure, or game-engine mesh streaming exists yet.
- The sparse-vs-dense comparison is diagnostic only and does not prove accuracy.
