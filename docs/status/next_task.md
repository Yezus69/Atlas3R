Phase 5H - Measured 3D Scene/Object Ingestion And Mesh Evaluation.

Goal: add a measured 3D scene/object dataset path and evaluate Atlas3R geometry,
mesh, and object outputs against measured references. This is not another
depth-only pseudo-label phase.

Start from the Phase 5G branch/report. Reload `README.md`, `PLANS.md`,
`docs/08_API_CONTRACTS.md`, `docs/status/progress.md`,
`docs/status/decisions.md`, `docs/status/active_task.md`, and
`docs/status/phase5g_multisequence_pose_odometry_report.md`.

Constraints:

- Do not claim mapping-ready, realtime, benchmark accuracy, or millimeter
  accuracy without an explicit evaluation report.
- Do not hallucinate completed or hidden geometry as measured geometry.
- Do not vendor third-party datasets, repos, model weights, or generated run
  artifacts.
- Preserve coordinate conventions, tensor shapes, teacher-signal cache schema,
  and runtime output contracts unless `docs/08_API_CONTRACTS.md` and tests are
  updated first.

Suggested first vertical slice:

1. Pick one locally available measured 3D scene/object dataset path or add a
   dependency-safe adapter stub plus local ingest path if no dataset is present.
2. Define concise contracts for measured mesh/object references and evaluation
   outputs.
3. Run the existing teacher/student depth plus pose path into a mesh/object
   diagnostic output on a small measured scene.
4. Report mesh metrics such as Chamfer/F-score thresholds and object metrics
   such as AP/IoU only where measured references exist.
5. Keep all outputs diagnostic and record the exact commands and gaps.
