# Codex Prompt - Atlas3R Phase 0D: CPU TSDF Reference Integrator on Synthetic Cube-Room

You are working in the existing public repo `Yezus69/Atlas3R` after Phase 0C.
Read `AGENTS.md`, `README.md`, `PLANS.md`, `docs/08_API_CONTRACTS.md`, and
the current `docs/status/*` files before coding. Then read only the source and
tests needed for the TSDF smoke path.

## Task goal

Implement a tiny deterministic CPU TSDF reference integrator for the Phase 0B
synthetic cube-room session. This is a correctness reference only, not the
real-time mapper.

Do **not** implement CUDA, neural models, teacher adapters, nvblox integration,
OpenGL, web servers, notebooks, trimesh, or GLB export in this task.

## Required implementation

1. Add a small CPU TSDF module under `src/atlas3r/mapping/`.
   - Integrate the Phase 0B analytic depth arrays and Phase 0A/0B
     `T_world_camera` poses into a voxel grid.
   - Use meters, the documented camera convention, and explicit
     `T_world_camera` naming.
   - Store deterministic per-voxel weight/confidence or uncertainty values.
   - Keep the implementation intentionally small and pure NumPy.

2. Extract a minimal deterministic surface result sufficient for tests.
   - A simple occupied-surface point cloud or coarse triangle/face
     approximation is acceptable for Phase 0D.
   - Preserve metadata: source frame IDs, voxel size, coordinate frame,
     metric scale source, observed coverage estimate, and uncertainty summary.

3. Compare the fused result against the synthetic ground-truth mesh with simple
   deterministic metrics.
   - Keep metrics conservative and clearly named.
   - Do not claim millimeter accuracy.
   - Report known limitations such as voxel resolution and analytic fixture
     assumptions.

4. Add CLI smoke command:

```bash
atlas3r smoke tsdf-cube-room --output <folder>
```

The command should generate or reuse a synthetic cube-room session, run the CPU
TSDF reference integration, write the minimal surface/metrics outputs under the
requested folder, and print concise paths.

5. Add tests.
   - TSDF integration is deterministic across runs.
   - The output carries uncertainty/confidence metadata.
   - The fused surface overlaps the synthetic room/object bounds within a
     voxel-scale tolerance.
   - The CLI smoke command succeeds.
   - Existing Phase 0A-0C tests keep passing.

6. Update docs/status.
   - Rewrite `docs/status/active_task.md` before coding with a concise Phase 0D
     plan and checklist.
   - Append results to `docs/status/progress.md` after verification.
   - Append to `docs/status/decisions.md` only if an interface or format
     decision changed.
   - Replace this file with the Phase 0E prompt before declaring done.

## Verification commands

Run:

```bash
python -m ruff format src tests
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m unittest discover -s tests -p 'test_*.py'
```

If `make` is available, also run:

```bash
make test
make lint
make typecheck
make smoke
make inspect
```

Record any unavailable command with the exact environment reason in
`docs/status/progress.md`.
