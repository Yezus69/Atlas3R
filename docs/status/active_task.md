# Active task

Codex should keep the current task checklist here so context compaction does not lose state.

```text
Goal: Phase 1E - add a dependency-free MeshChunk sidecar writer for CPU TSDF
replay outputs, preserving observed-only confidence/uncertainty metadata and
explicit low-fidelity/reference-only flags.
Relevant docs read: AGENTS.md, README.md, PLANS.md, docs/08_API_CONTRACTS.md,
docs/status/active_task.md, docs/status/progress.md,
docs/status/decisions.md, docs/status/next_task.md.
Plan:
1. Inspect only the MeshChunk contract/validation code, CPU TSDF smoke and
   teacher-cache TSDF replay writers, CLI smoke commands, and focused tests.
2. Add a deterministic TSDF surface-to-MeshChunk helper that consumes
   TSDFSurface artifacts and emits an observed, low-fidelity triangle sidecar
   with source frame IDs, coordinate frame, voxel size, scale source, observed
   coverage, confidence, and mean/p95 uncertainty metadata.
3. Wire the CPU TSDF smoke commands to optionally write the sidecar under the
   requested output directory without removing or renaming existing artifacts.
4. Add focused tests for contract validation, deterministic fixture replay
   sidecars, metadata preservation, and path-named failures for missing or
   malformed surface artifacts.
5. Update docs/08_API_CONTRACTS.md and status logs, append a decision only if
   the public sidecar format changes an interface, replace next_task.md with
   the Phase 2A prompt, and run the requested verification commands.
Checklist:
- [x] Read required docs and status files.
- [x] Inspect focused source/tests.
- [x] Add TSDF surface MeshChunk sidecar helper.
- [x] Wire optional CLI smoke output.
- [x] Add/update focused tests.
- [x] Update API/status docs and Phase 2A handoff.
- [x] Run verification commands and record results.
Known exclusions: No GLB/PLY/trimesh/marching-cubes dependencies, model
downloads, vendored third-party code, cloud APIs, neural inference, CUDA, or
real-world accuracy claims in Phase 1E.
```
