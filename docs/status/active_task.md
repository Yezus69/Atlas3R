# Active task

Goal: Phase 4B checkpoint inference bridge from Phase 4A
`checkpoint_last.pt` into `DepthObservation` and CPU TSDF smoke artifacts.

Checklist:

- [x] Read Phase 4B prompt, status docs, API contracts, training, mapper, TSDF,
  frame-source, and CLI files.
- [x] Add a dependency-safe tiny checkpoint loader with clear missing-Torch errors.
- [x] Add inference helpers for `StudentClipInput` and ordered `FramePacket`
  inputs.
- [x] Convert predictions to validated `DepthObservation` records with
  uncertainty, confidence, coordinate frame, scale source, and truth-boundary
  diagnostics.
- [x] Add a deterministic synthetic checkpoint-to-TSDF smoke helper/CLI with
  predicted-vs-target metrics and preview artifacts.
- [x] Add optional Torch tests that skip when the `train` extra is unavailable.
- [x] Update API/status docs within context budgets.
- [x] Run relevant lint, typecheck, tests, smoke, and diff checks.

Exclusions: no external teacher models, real datasets, video decoding, runtime
scheduler changes, TSDF internals, GLB/PLY export, web servers, notebooks, or
broad inspection bundles.
