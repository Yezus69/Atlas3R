Phase 5G - Pose Tracking Teacher and Learned Pose Runtime.

Goal: add a real external pose-tracking teacher path, such as VGGT/LingBot/other
local pose pseudo-label output, without vendoring third-party code or weights.
Generate validated pose-aware teacher signals, inspect pose/depth/map quality,
train the temporal student to reduce dependence on source clip-cache poses, and
compare `student-relative` runtime behavior against Phase 5E/5F.

Start from the Phase 5F branch/report. Reload `README.md`, `PLANS.md`,
`docs/08_API_CONTRACTS.md`, `docs/status/progress.md`,
`docs/status/decisions.md`, `docs/status/active_task.md`, and
`docs/status/phase5f_real_depthpro_mixed_training_report.md`.

Constraints:

- Do not fake pose teacher outputs or relabel source TUM poses as external pose
  pseudo-labels.
- Keep external models/checkpoints outside the Atlas3R repo behind dependency
  safe adapters or local-ingest paths.
- Keep truth boundaries explicit: no mapping-ready, realtime, accuracy, or
  benchmark claims without a report that proves them.
- Preserve the stable teacher-signal/runtime contracts or update
  `docs/08_API_CONTRACTS.md` and tests first.

Suggested first vertical slice:

1. Preflight available local pose-teacher packages/checkpoints or local output
   folders.
2. Add the smallest validated external pose teacher ingest/run path.
3. Inspect pose deltas against TUM source poses without treating source poses as
   external teacher truth.
4. Train a bounded measured-depth plus pose-pseudo temporal student.
5. Run `runtime stream-student-map --pose-mode both` and compare Phase 5G
   `student-relative` behavior against the Phase 5E/5F reports.
