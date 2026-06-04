# Active Task - Phase 6A Product-Slice Mapper Recording Mesh

Goal: add a validated Atlas3R recording format, import measured TUM/clip-cache
streams, fuse recording depth+pose through CPU TSDF in chronological order, and
export inspectable point-cloud/optional real mesh artifacts with diagnostic
latency, memory, and quality reports.

Branch: `codex/phase6a-product-slice-mapper-recording-mesh`

Checklist:

- [x] Start from `codex/phase5h-vggt-pose-pointmap-teacher` and create the
  Phase 6A branch.
- [x] Read the goal file, API contracts, status docs, TUM/clip-cache code,
  CPU TSDF path, and runtime report/export helpers.
- [x] Add dependency-light recording schema, frame validation, and
  `atlas3r recording validate`.
- [x] Add `recording from-tum` and `recording from-clip-cache` importers.
- [x] Add measured recording-to-`DepthObservation` streaming fusion runtime.
- [x] Export `surface_points.ply`, optional real marching-cubes mesh, TSDF
  artifacts, preview HTML, runtime events, and reports.
- [x] Add focused tests for validation, path traversal, importers, runtime,
  PLY, optional mesh behavior, CLI help, and status handoff.
- [x] Run required format, lint, typecheck, unit, diff, and available make
  verification commands.
- [x] Run real TUM evidence if local data exists; otherwise record the
  missing-data blocker and fixture evidence.
- [x] Update compact progress, decisions, API contracts, Phase 6A report, and
  next task.
