# Codex Prompt - Atlas3R Phase 4B: Trained Checkpoint Inference Bridge to DepthObservation and TSDF Smoke

You are working in `Yezus69/Atlas3R` after Phase 4A.

## Goal

Load the Phase 4A `checkpoint_last.pt`, run the tiny trained model on
synthetic or NPZ `FramePacket` inputs, convert predictions into the existing
`DepthObservation` mapper boundary, feed the CPU TSDF smoke path, and compare
predicted-vs-target TSDF/preview artifacts.

This is a checkpoint-inference-to-mapper bridge. It should prove that the first
training MVP can drive the existing geometry-facing contracts without changing
runtime scheduling or TSDF internals.

## Scope

- Add a dependency-safe checkpoint loader that fails clearly when Torch is
  missing.
- Add a small inference helper for `TinyDepthPoseNet` over existing
  `FramePacket` / `StudentClipInput` data.
- Convert model outputs to `DepthObservation` with explicit uncertainty,
  confidence, coordinate frame, metric scale source, and truth-boundary
  metadata.
- Add a CPU TSDF smoke command or focused helper that consumes those predicted
  observations on deterministic synthetic inputs.
- Write compact predicted-vs-target metrics/preview artifacts sufficient for
  debugging the bridge.
- Add tests that skip Torch-dependent inference when the optional train extra is
  unavailable.

## Exclusions

Do not add external teacher models, real datasets, video decoding, runtime
scheduler changes, GLB/PLY export, object fusion, web servers, notebooks, or a
new broad inspection bundle.

## Done criteria

- Base imports remain Torch-free.
- Missing Torch produces clear installation errors.
- Loaded checkpoints preserve Phase 4A truth-boundary flags.
- Predicted observations validate as `DepthObservation`.
- CPU TSDF smoke accepts predicted observations and records uncertainty.
- Tests and status docs are updated within context budgets.
