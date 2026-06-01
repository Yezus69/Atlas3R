# Active task

Codex should keep the current task checklist here so context compaction does not lose state.

```text
Goal: Phase 2A - add a dependency-free WorldMap sidecar path for CPU TSDF
MeshChunk sidecars, preserving observed-only truth boundaries and
confidence/uncertainty metadata.
Relevant docs read: AGENTS.md, README.md, PLANS.md, docs/08_API_CONTRACTS.md,
docs/status/active_task.md, docs/status/progress.md,
docs/status/decisions.md, docs/status/next_task.md.
Plan:
1. Inspect only the WorldMap/MeshChunk contracts, Phase 1E MeshChunk sidecar
   helper, CPU TSDF smoke/replay writers, CLI smoke/inspect wiring, and focused
   synthetic tests.
2. Add a deterministic WorldMap sidecar helper that consumes
   mesh_chunk_sidecar.json, validates the MeshChunk contract, wraps it in a
   contract-valid WorldMap with no object meshes, and preserves coordinate
   frame, source frames, voxel size, scale source, coverage, confidence, and
   mean/p95 uncertainty metadata.
3. Wire optional CPU TSDF smoke/replay output and a validation-oriented inspect
   command without removing or renaming existing Phase 0D/1D/1E artifacts.
4. Add focused tests for contract validation, deterministic fixture replay map
   sidecars, metadata preservation, and path-named missing/malformed MeshChunk
   sidecar errors.
5. Update docs/08_API_CONTRACTS.md and status logs, append a decision for the
   public sidecar format, replace next_task.md with the Phase 2B prompt, and
   run the requested verification commands.
Checklist:
- [x] Read required docs and status files.
- [x] Inspect focused source/tests.
- [x] Add TSDF MeshChunk WorldMap sidecar helper.
- [x] Wire optional CLI smoke/inspect output.
- [x] Add/update focused tests.
- [x] Update API/status docs and Phase 2B handoff.
- [x] Run verification commands and record results.
Known exclusions: No GLB/PLY/trimesh/marching-cubes dependencies, model
downloads, vendored third-party code, cloud APIs, neural inference, CUDA, or
real-world accuracy claims in Phase 2A.
```
