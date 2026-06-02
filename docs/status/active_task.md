# Active task

Goal: Phase 4D TUM RGB-D checkpoint evaluation, predicted-vs-target TSDF
diagnostics, block validation split, and one targeted TinyMetricDepthNetV2
training path.

Checklist:

- [x] Read Phase 4D goal, compact status docs, API contracts, current TUM
  training/checkpoint/mapping code, and relevant tests.
- [x] Create branch `codex/phase4d-tum-eval-v2-training` from
  `codex/overnight-realdata-tum-rgbd`.
- [x] Add `atlas3r eval tum-rgbd-checkpoint` with depth, pose-center, preview,
  sample, trajectory, and truth-boundary metadata outputs.
- [x] Add optional `--write-tsdf` predicted-vs-target CPU TSDF diagnostics and
  conservative NumPy point-set metrics.
- [x] Add TUM manifest `--split-policy every10|block` with block-tail
  validation and manifest metadata.
- [x] Add exactly one v2 model path using RGB plus camera ray channels, and keep
  v1 checkpoint compatibility.
- [x] Add masked log-depth loss option and CLI flags for `--model` and
  `--depth-loss`.
- [x] Add focused unit/CLI tests and update API/status docs within context
  budgets.
- [x] Run required format, lint, typecheck, unittest, and diff-check gates.
- [ ] Commit code only, then run real baseline evaluation/training/v2 evaluation
  if CUDA/data/checkpoints are available.
- [ ] Write and commit compact Phase 4D report with commands, metrics, map
  diagnostics, blockers, and next task.

Exclusions: no external teacher models, OpenCV, DDP, notebooks, web servers,
mesh export, base dependency additions, or accuracy/performance claims.
