# Active task

Codex should keep the current task checklist here so context compaction does not lose state.

```text
Goal: Phase 2B - add a dependency-free CPU TSDF output folder inspector that
validates Phase 0D/1D surface artifacts, Phase 1E MeshChunk sidecars, and Phase
2A WorldMap sidecars together, then emits deterministic JSON for early mapper
pipeline checks.
Relevant docs read: AGENTS.md, README.md, PLANS.md, docs/08_API_CONTRACTS.md,
docs/status/active_task.md, docs/status/progress.md,
docs/status/decisions.md, docs/status/next_task.md.
Plan:
1. Inspect only the CPU TSDF surface artifact loaders, MeshChunk sidecar helper,
   WorldMap sidecar helper, CLI inspect wiring, Makefile targets, and focused
   synthetic tests.
2. Add a narrow output-folder inspector that validates metadata.json,
   surface_points.npz, optional metrics.json, mesh_chunk_sidecar.json, and
   world_map_sidecar.json, with explicit required-sidecar modes and
   path-named mismatch errors.
3. Cross-check coordinate frame, source frame IDs, voxel size, metric scale
   source, observed coverage, confidence summary, and mean/p95 uncertainty
   across the surface, MeshChunk, and WorldMap records.
4. Wire `atlas3r inspect tsdf-output --input <folder>` as deterministic JSON
   without adding dependencies or changing existing smoke artifact names.
5. Add focused synthetic tests for TSDF smoke and teacher-cache replay
   inspection determinism, metadata mismatch rejection, required sidecar
   errors, and CLI behavior.
6. Update docs/08_API_CONTRACTS.md, status logs, next_task.md for Phase 2C, and
   run the requested verification commands.
Checklist:
- [x] Read required docs and status files.
- [x] Inspect focused source/tests.
- [x] Add TSDF output folder inspector.
- [x] Wire CLI inspect output.
- [x] Add/update focused tests.
- [x] Update API/status docs and Phase 2C handoff.
- [x] Run verification commands and record results.
Known exclusions: No GLB/PLY/trimesh/marching-cubes dependencies, model
downloads, vendored third-party code, cloud APIs, neural inference, CUDA, or
real-world accuracy claims in Phase 2B.
```
