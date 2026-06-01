# Codex Prompt - Atlas3R Phase 2C: CPU TSDF Inspection Regression Bundle

You are working in the existing public repo `Yezus69/Atlas3R` after Phase 2B.
Read `AGENTS.md`, `README.md`, `PLANS.md`, `docs/08_API_CONTRACTS.md`, and
the current `docs/status/*` files before coding. Then read only the source and
tests needed for CPU TSDF output inspection, deterministic smoke outputs,
fixture teacher-cache replay, and status handoff behavior.

## Task goal

Add a dependency-free regression bundle path for CPU TSDF inspection outputs.
The bundle should capture deterministic inspection JSON from the synthetic TSDF
smoke path and the fixture teacher-cache replay path, record artifact hashes for
the validated JSON/NPZ sidecars, and make it easy to compare early mapper
pipeline outputs without adding mesh export dependencies. Do not add
GLB/PLY/trimesh/marching-cubes dependencies, download model weights, vendor
third-party repositories, call cloud APIs, run neural inference, or claim
real-world accuracy.

## Required implementation

1. Add a CPU TSDF inspection regression bundle writer.
   - Generate or consume complete CPU TSDF output folders with
     `metadata.json`, `surface_points.npz`, optional `metrics.json`,
     `mesh_chunk_sidecar.json`, and `world_map_sidecar.json`.
   - Run the Phase 2B folder inspector in `complete` mode and write
     deterministic inspection JSON into the bundle.
   - Record deterministic SHA-256 hashes and file sizes for the validated
     JSON/NPZ artifacts without embedding large array payloads in JSON.
   - Keep truth-boundary flags explicit: low-fidelity/reference-only,
     observed-only, not completed, and not an accuracy report.

2. Wire CLI output.
   - Add or extend an `atlas3r inspect ...` command that writes the regression
     bundle to a requested output directory.
   - The command must be dependency-free and deterministic.
   - Do not remove or rename existing Phase 0D/1D/1E/2A/2B artifacts.

3. Add validation and tests.
   - Synthetic TSDF smoke and fixture teacher-cache replay bundles are
     deterministic.
   - Hash manifests reject missing or modified required artifacts with
     path-named errors.
   - Existing Phase 0A-2B tests keep passing.

4. Update docs/status.
   - Rewrite `docs/status/active_task.md` before coding with a concise Phase 2C
     plan and checklist.
   - Update `docs/08_API_CONTRACTS.md` for any public bundle output or CLI
     behavior.
   - Append results to `docs/status/progress.md` after verification.
   - Append to `docs/status/decisions.md` only if an interface or format
     decision changed.
   - Replace this file with the Phase 2D prompt before declaring done.

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
