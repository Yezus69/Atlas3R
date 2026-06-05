Offline V0.6: plug the first real geometry witness into the existing vertical
world-builder tracer.

Start from the latest Offline V0.5 tracer branch. Reload `README.md`,
`PLANS.md`, `docs/00_PROJECT_OBJECTIVE.md`,
`docs/01_OFFLINE_WORLD_BUILDER_ARCHITECTURE.md`,
`docs/02_DATA_AND_TRUTH_BOUNDARY.md`,
`docs/03_TEACHER_CONSENSUS_AND_OPTIMIZATION.md`,
`docs/04_API_CONTRACTS.md`, `docs/05_EVALUATION_AND_QUALITY.md`, and
`docs/status/*`.

Goal: choose one dependency-safe real geometry witness path, preferably VGGT or
Depth Pro, and wire it through `python -m atlas3r offline build-world` so the
same command changes its report from "geometry unavailable/debug only" to
"teacher-proposed geometry available" when the external dependency is present.

Rules:

- Continue by vertical slices across ingestion, proposal cache, camera/scale
  ledger, consensus world state, geometry preview, diagnostics, quality report,
  and training-cache manifest.
- Do not implement a single-stage module that is unused by `offline build-world`
  or by tests feeding the same pipeline.
- Keep the adapter isolated under `src/atlas3r/models/adapters/` and make
  missing dependencies explicit failure points, not import-time crashes.
- Teacher geometry remains a proposal, not measured truth. Physical accuracy
  claims require anchors or a named evaluation report.

Acceptance:

- `import atlas3r` remains free of heavy optional dependencies.
- `offline build-world` still produces the full V0.5 artifact tree when the
  witness is unavailable.
- With the chosen witness available on a tiny smoke input, the proposal cache,
  world state, geometry preview, render diagnostics, and quality report all
  reference teacher-proposed geometry.
- Reports still mark unanchored RGB output as not physically accurate and not
  training-quality.
- Focused tests and the standard format, lint, typecheck, unit, CLI, smoke, and
  diff checks pass.
