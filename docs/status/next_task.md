Phase 6I - Train Tiny Student From Measured And Teacher Temporal Caches

Goal: use the Phase 6H validated teacher temporal cache as pseudo-label
training data for a tiny SMGT-style student, mix it with measured temporal
caches when available, and produce a diagnostic RGB student mapping run without
claiming RGB-only readiness.

Start from branch `codex/phase6h-teacher-stitch-cache`. Reload `README.md`,
`PLANS.md`, `docs/08_API_CONTRACTS.md`, `docs/09_EVALUATION.md`,
`docs/status/progress.md`, `docs/status/decisions.md`,
`docs/status/active_task.md`, and
`docs/status/phase6h_teacher_stitch_cache_report.md`.

Context:

- Phase 6H proved real VGGT RGB teacher windows can be stitched with Sim3 and
  exported as validated `atlas3r_teacher_temporal_cache` clips.
- The real stitched Freiburg cache lives under
  `runs/phase6h_rgb_teacher_stitched_freiburg1_xyz_val/teacher_temporal_cache`
  when local run artifacts are present.
- Teacher cache labels are pseudo labels only. They must retain lower weights
  than measured labels and must not become measured mapping truth.
- Phase 6H did not train a student and did not prove realtime, metric-scale,
  loop closure, object-aware fusion, hidden-geometry completion, or RGB-only
  student mapping readiness.

Required work:

- Add a tiny training data adapter that can mix existing measured temporal
  caches and Phase 6H `TeacherTemporalCacheDataset` samples with explicit
  per-target weights.
- Train or overfit the smallest dependency-safe `smgt_tiny` student slice on a
  fixture plus one real teacher-cache clip subset, recording losses, config,
  checkpoint metadata, truth flags, and source cache hashes/paths.
- Add an inference bridge that converts tiny student RGB predictions into the
  existing pseudo `DepthObservation` mapper path with explicit student and
  pseudo truth flags.
- Run a diagnostic map on a tiny real sequence and compare against the Phase 6H
  teacher-cache target or eval-only source measurements without calling it an
  accuracy report.
- Update contracts, evaluation notes, status docs, and a Phase 6I report.

Acceptance:

- Unit tests cover mixed measured/pseudo sample weighting, cache loading,
  checkpoint metadata, and inference-to-mapper shape/truth contracts.
- A fixture cache train/infer/map path runs without optional VGGT dependencies.
- A real small-cache train or overfit command completes locally, or the phase is
  explicitly marked blocked with exact dependency/hardware failures.
- Verification includes format, lint, typecheck, focused tests, full unittest
  discovery, `git diff --check`, and the real train/infer/map evidence command.
