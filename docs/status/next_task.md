# Codex Prompt - Atlas3R Phase 4C: Coherent Checkpoint Inference Sequence Smoke

You are working in `Yezus69/Atlas3R` after Phase 4B.

## Goal

Extend the Phase 4B checkpoint-inference bridge from one-frame procedural
debug inputs to a deterministic coherent multi-frame synthetic sequence, then
measure depth, camera-center, and TSDF consistency across the sequence.

This should stay a diagnostic bridge for the tiny Phase 4A model. It must not
claim real-capture performance or benchmark accuracy.

## Scope

- Add a small coherent synthetic RGB/depth sequence generator that shares one
  static scene across frames and varies only camera center within the existing
  coordinate convention.
- Convert that sequence through `StudentClipInput`, run the Phase 4A checkpoint,
  and emit validated `DepthObservation` records for every frame.
- Feed all predicted observations into the existing CPU TSDF smoke helper and
  compare against the sequence target TSDF on the same grid.
- Record compact per-frame and aggregate depth/camera-center/TSDF metrics.
- Keep base imports Torch-free and skip Torch-dependent tests when the optional
  train extra is unavailable.

## Exclusions

Do not add real datasets, video decoding, external teacher models, runtime
scheduler changes, TSDF internals, GLB/PLY export, notebooks, web servers, or
broad inspection bundles.

## Done Criteria

- Coherent sequence target and prediction observations validate as
  `DepthObservation`.
- Predicted TSDF smoke records source frame IDs, uncertainty, confidence,
  coordinate frame, metric scale source, and truth-boundary metadata.
- Metrics distinguish diagnostic smoke output from accuracy reports.
- API/status docs and tests are updated within context budgets.
- Relevant lint, typecheck, unit tests, and smoke commands are run or explicitly
  documented as unavailable.
