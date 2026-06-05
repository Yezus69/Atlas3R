# Active Task - Phase 6F Live Mesh Chunk Updates

Goal: turn measured Phase 6E sparse replay into observed-only live triangle mesh
chunk updates with stable chunk IDs, versioned NPZ/PLY artifacts, profiling
evidence, tests, and a committed result.

Branch: `codex/phase6f-live-mesh-chunks`

Checklist:

- [x] Confirm clean worktree and create the Phase 6F branch from Phase 6E.
- [x] Read the required architecture, contract, evaluation, status, runtime,
  sparse mapper, CLI, and regression test files.
- [x] Extend sparse TSDF updates with dirty block coordinates and per-stage
  timings without changing existing runtime behavior.
- [x] Add dependency-safe observed mesh chunk schemas, fallback sparse meshing,
  NPZ/PLY artifact writers, manifest, and update log.
- [x] Integrate bounded incremental mesh chunk updates into
  `runtime live-replay-recording` behind new CLI flags.
- [x] Add focused schema, dirty-block, mesher, artifact loading, live replay,
  and CLI regression tests.
- [x] Run required format, lint, typecheck, unit, diff, and focused commands.
- [x] Run real/largest measured replay mesh evidence and Phase 6F profiles.
- [x] Update concise README, PLANS, architecture, API contracts, evaluation,
  progress, decisions, next task, and Phase 6F report docs.
- [x] Commit with `feat(mapping): stream observed mesh chunks from sparse replay`.
