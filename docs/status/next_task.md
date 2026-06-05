Phase 6G - Accelerate Sparse Mapping And Mesh Chunk Updates

Goal: reduce measured replay map+mesh update latency enough that observed-only
mesh chunks become usable for a live preview loop, without weakening the Phase
6F truth boundary.

Start from branch `codex/phase6f-live-mesh-chunks`. Reload `README.md`,
`PLANS.md`, `docs/08_API_CONTRACTS.md`, `docs/status/progress.md`,
`docs/status/decisions.md`, `docs/status/active_task.md`, and
`docs/status/phase6f_live_mesh_chunks_report.md`.

Context:

- Phase 6F exports nonzero loadable observed mesh chunks from measured replay.
- The optimized preview profile still has map+mesh p95 above 33 ms, with sparse
  candidate generation and mesh chunk update/export latency as the main
  bottlenecks.
- No realtime, RGB-only, hidden-geometry, object-aware fusion, or accuracy claim
  exists.

Required work:

- Profile sparse candidate generation, projection, apply updates, snapshot
  creation, fallback meshing, NPZ/PLY writing, and JSON event writing separately.
- Implement at least one measured acceleration that preserves deterministic
  dirty block IDs, observed-only payloads, queue bounds, and existing CLI/API
  contracts.
- Prefer dependency-safe NumPy/vectorization/cache improvements first. Optional
  Torch/CUDA paths may be added only behind clear availability checks and must
  not be required for unit tests.
- Keep `runtime fuse-recording` backends and `runtime live-replay-recording`
  without mesh flags working.
- Add focused tests for any changed scheduling, caching, or backend behavior.
- Run format, lint, typecheck, unit tests, diff check, and measured replay
  profiles for baseline-compatible, preview/live-friendlier, and coarse-live
  settings.

Acceptance:

- Preserve nonzero loadable NPZ/PLY mesh chunks from the measured replay.
- Either get preview/live-friendlier map+mesh p95 below 33 ms on this machine,
  or document a measured latency reduction of at least 25% versus the Phase 6F
  optimized preview profile and name the remaining bottleneck with numbers.
- Update concise docs/status and commit the result.
