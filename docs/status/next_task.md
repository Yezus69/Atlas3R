# Codex Prompt - Atlas3R Phase 2B: CPU TSDF Output Folder Map Inspection

You are working in the existing public repo `Yezus69/Atlas3R` after Phase 2A.
Read `AGENTS.md`, `README.md`, `PLANS.md`, `docs/08_API_CONTRACTS.md`, and
the current `docs/status/*` files before coding. Then read only the source and
tests needed for CPU TSDF output artifacts, MeshChunk sidecars, WorldMap
sidecars, and CLI inspect behavior.

## Task goal

Add a dependency-free inspection path for complete CPU TSDF output folders. The
path should validate Phase 0D/1D surface artifacts, the Phase 1E MeshChunk
sidecar, and the Phase 2A WorldMap sidecar together, then emit deterministic
inspection JSON for early mapper pipeline checks. Do not add GLB/PLY/trimesh/
marching-cubes dependencies, download model weights, vendor third-party
repositories, call cloud APIs, run neural inference, or claim real-world
accuracy.

## Required implementation

1. Add a CPU TSDF output folder inspector.
   - Validate `metadata.json`, `surface_points.npz`, optional `metrics.json`,
     `mesh_chunk_sidecar.json`, and `world_map_sidecar.json` when present.
   - Cross-check coordinate frame, source frame IDs, voxel size, metric scale
     source, observed coverage estimate, confidence summary, and mean/p95
     uncertainty across surface, MeshChunk, and WorldMap metadata.
   - Keep truth-boundary flags explicit: low-fidelity/reference-only,
     observed-only, not completed, and not an accuracy report.
   - Report missing required artifacts with path-named errors.

2. Wire CLI inspect output.
   - Add or extend an `atlas3r inspect ...` command that reads a CPU TSDF output
     directory and prints deterministic JSON.
   - The command must be dependency-free and deterministic.
   - Do not remove or rename existing Phase 0D/1D/1E/2A artifacts.

3. Add validation and tests.
   - Fixture TSDF smoke and teacher-cache replay outputs inspect deterministically.
   - Cross-artifact metadata mismatches are rejected with path-named errors.
   - Missing MeshChunk or WorldMap sidecars are explicit when required by the
     selected inspect mode.
   - Existing Phase 0A-2A tests keep passing.

4. Update docs/status.
   - Rewrite `docs/status/active_task.md` before coding with a concise Phase 2B
     plan and checklist.
   - Update `docs/08_API_CONTRACTS.md` for any public inspect output or CLI
     behavior.
   - Append results to `docs/status/progress.md` after verification.
   - Append to `docs/status/decisions.md` only if an interface or format
     decision changed.
   - Replace this file with the Phase 2C prompt before declaring done.

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
