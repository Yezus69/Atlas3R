# Active Task

Goal: Phase 5G multi-sequence measured TUM pose/depth supervision, SE(3)
temporal student training, and student-odometry runtime diagnostics.

Branch: `codex/phase5g-multisequence-pose-odometry`

Checklist:

- [x] Start from `codex/phase5f-real-depthpro-mixed-training-runtime` and create
  the Phase 5G branch.
- [x] Read the Phase 5G goal plus compact architecture, contract, and status
  docs.
- [x] Preflight local external pose-teacher environment variables.
- [x] Extend TUM RGB-D sequence specs and paired URL override support.
- [x] Harden measured teacher-signal pose metadata and full-SE(3) validation.
- [x] Add one SE(3)-capable temporal student path with full relative rotation.
- [x] Add SE(3) losses and pose rollout metrics.
- [x] Fix CUDA AMP rotation-loss stability so optimizer steps are not skipped.
- [x] Add `student-odometry` runtime pose mode and diagnostic pose reports.
- [x] Add focused tests for dataset specs, pose contracts, losses, AMP, and
  runtime odometry.
- [x] Run available real-data training/evaluation on local measured
  `freiburg1_xyz` caches.
- [x] Write the Phase 5G report and update compact status files.
- [x] Run final format, lint, typecheck, unittest, diff, and make checks when
  available.
- [x] Commit code/docs only after tests pass.

Known constraints:

- No local external VGGT/LingBot pose teacher outputs were configured; none were
  fabricated.
- Only `freiburg1_xyz` was present locally for the real run; additional TUM
  sequence specs are implemented but not locally trained yet.
- Generated datasets, caches, checkpoints, `.npz`, PLY, previews, run folders,
  model weights, and external repos stay uncommitted.
- All outputs remain diagnostic unless an explicit evaluation report supports
  stronger claims.
