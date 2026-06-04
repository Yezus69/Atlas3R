Phase 5H - External Pose/Pointmap Teacher Generation Before Mesh/Object Work.

Goal: improve pose and geometry supervision before attempting measured
mesh/object ingestion. Phase 5G.1 showed that multi-sequence measured TUM
training works, but student odometry remains too drifty and depth generalization
is uneven, especially on `freiburg1_desk`.

Start from branch `codex/phase5g1-multisequence-tum-generalization`. Reload
`README.md`, `PLANS.md`, `docs/08_API_CONTRACTS.md`,
`docs/status/progress.md`, `docs/status/decisions.md`,
`docs/status/active_task.md`, and
`docs/status/phase5g1_multisequence_tum_generalization_report.md`.

Constraints:

- Do not claim mapping-ready, realtime, benchmark accuracy, or millimeter
  accuracy without an explicit evaluation report.
- Do not hallucinate completed or hidden geometry as measured geometry.
- Do not vendor third-party datasets, repos, model weights, or generated run
  artifacts.
- Preserve coordinate conventions, tensor shapes, teacher-signal cache schema,
  and runtime output contracts unless `docs/08_API_CONTRACTS.md` and tests are
  updated first.
- Keep third-party pose/pointmap models isolated behind dependency-safe
  adapters or local-output ingestion paths.

Suggested first vertical slice:

1. Pick VGGT or LingBot-Map based on what is locally available, or add only a
   dependency-safe runner/local-ingest blocker report if neither is configured.
2. Generate or ingest external pose/pointmap teacher outputs for the Phase 5G.1
   TUM validation/train sequences without committing model weights or caches.
3. Validate the external outputs into existing teacher-signal cache contracts,
   with explicit pseudo-label and truth-boundary metadata.
4. Train or evaluate the existing temporal student against the stronger teacher
   signal on a small multi-sequence slice.
5. Compare student-odometry drift against the Phase 5G.1 report before starting
   measured mesh/object work.
