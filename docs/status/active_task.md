# Active task

Goal: Phase 4A training MVP on deterministic procedural RGB + metric-depth
synthetic samples.

Checklist:

- [x] Ignore stale Phase 3D `next_task.md` as an execution source.
- [x] Read the Phase 4A prompt, API/status docs, and focused student/data/CLI files.
- [x] Add optional `train` dependency extra with dependency-safe Torch helpers.
- [x] Add deterministic procedural RGB/depth samples and `StudentClipInput` conversion.
- [x] Add tiny trainable Torch depth/uncertainty/confidence/camera-center model.
- [x] Add finite supervised depth, sigma NLL, confidence, and camera-center losses.
- [x] Add `atlas3r train synthetic-overfit` writing config, metrics, checkpoint,
  summary, prediction sample, and dependency-free HTML/SVG preview.
- [x] Add import-safety, dataset, and optional Torch training smoke tests.
- [x] Update API/status docs and Phase 4B handoff within context budgets.
- [x] Run Ruff, mypy, unittest, diff check, optional training smoke, and make if available.

Exclusions: no external teacher models, real datasets, video decoding, TSDF,
runtime, GLB/PLY export, web servers, notebooks, or inspection bundles.
