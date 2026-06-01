# Codex Prompt - Atlas3R Phase 2A: CPU TSDF MeshChunk WorldMap Assembly

You are working in the existing public repo `Yezus69/Atlas3R` after Phase 1E.
Read `AGENTS.md`, `README.md`, `PLANS.md`, `docs/08_API_CONTRACTS.md`, and
the current `docs/status/*` files before coding. Then read only the source and
tests needed for CPU TSDF replay outputs, MeshChunk sidecars, WorldMap
contracts, and CLI smoke/inspect artifacts.

## Task goal

Add a dependency-free WorldMap assembly path for CPU TSDF replay outputs. The
path should load the Phase 1E MeshChunk sidecar, validate it against the
existing `MeshChunk` contract, wrap it in a contract-valid `WorldMap`, and write
a deterministic map sidecar suitable for early mapper pipeline tests. Do not add
GLB/PLY/trimesh/marching-cubes dependencies, download model weights, vendor
third-party repositories, call cloud APIs, run neural inference, or claim
real-world accuracy.

## Required implementation

1. Add a CPU TSDF WorldMap sidecar helper.
   - Consume Phase 1E `mesh_chunk_sidecar.json` outputs.
   - Emit a deterministic `world_map_sidecar.json` containing a contract-valid
     `WorldMap` with the observed MeshChunk and no invented object meshes.
   - Preserve coordinate frame, source frame IDs, voxel size, metric scale
     source, observed coverage estimate, and mean/p95 uncertainty in map
     metadata.
   - Keep all truth-boundary flags: low-fidelity/reference-only, observed-only,
     not completed, and not an accuracy report.

2. Wire CLI smoke/inspect output.
   - Extend the relevant CPU TSDF smoke or inspect command(s) to optionally
     write/validate the WorldMap sidecar under an output directory.
   - Keep the command dependency-free and deterministic.
   - Do not remove or rename existing Phase 0D/1D/1E artifacts.

3. Add validation and tests.
   - The WorldMap sidecar validates against the existing `WorldMap` and
     `MeshChunk` contracts.
   - Fixture replay writes deterministic map sidecars.
   - Mesh confidence and uncertainty metadata survive into map metadata.
   - Path-named failures are explicit when required MeshChunk sidecars are
     missing or malformed.
   - Existing Phase 0A-1E tests keep passing.

4. Update docs/status.
   - Rewrite `docs/status/active_task.md` before coding with a concise Phase 2A
     plan and checklist.
   - Update `docs/08_API_CONTRACTS.md` for any public map sidecar or CLI
     behavior.
   - Append results to `docs/status/progress.md` after verification.
   - Append to `docs/status/decisions.md` only if an interface or format
     decision changed.
   - Replace this file with the Phase 2B prompt before declaring done.

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
