# Active Task - Core SMGT Tiny Student Mapping

Goal: train a real learned `SMGTTiny` diagnostic student from the Phase 6H
teacher temporal cache and use its checkpoint for RGB-only sparse TSDF mesh
chunk mapping without running VGGT at student inference.

Checklist:

- [x] Confirm clean worktree, branch, required docs/code, cache, recording, and CUDA.
- [x] Add `atlas3r.models.smgt` config/model/memory/geometry/checkpoint package.
- [x] Add SMGT teacher-cache dataset adapter, losses, training, and eval helpers.
- [x] Register `python -m atlas3r train smgt-tiny`.
- [x] Add `runtime map-rgb-student` checkpoint inference to `DepthObservation` to sparse TSDF/mesh chunks.
- [x] Add model, loss, dataset, checkpoint, runtime, CLI, and regression tests.
- [x] Run overfit/debug training and normal weighted training on the Phase 6H cache.
- [x] Run real RGB student mapping on the Freiburg validation recording.
- [x] Update contracts, evaluation notes, progress, decisions, phase report, and next prompt.
- [x] Run required verification and commit only source/docs/tests.
