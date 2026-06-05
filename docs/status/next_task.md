Core Phase A3 - Replace Or Harden SMGT-Tiny Before Object Fusion

Goal: fix the student core and confidence calibration before any object/dynamic
fusion work. Do not add SAM, object-aware sparse fusion, or dynamic-scene fusion
until this phase passes heldout student-quality gates.

Start from branch `codex/core-smgt-tiny-generalization-gauntlet`. Reload
`README.md`, `PLANS.md`, `docs/01_SYSTEM_ARCHITECTURE.md`,
`docs/08_API_CONTRACTS.md`, `docs/09_EVALUATION.md`,
`docs/status/progress.md`, `docs/status/decisions.md`,
`docs/status/active_task.md`, and
`docs/status/core_smgt_tiny_a2_generalization_report.md`.

Context:

- Core A2 added strict temporal split manifests, heldout metrics, confidence/
  sigma/dynamic mapping gates, heldout RGB-only mapping, and a long-run replay.
- The A2 training run:
  `runs/core_smgt_tiny_a2_weighted_split_freiburg1_xyz_val`.
- `checkpoint_best.pt` and `checkpoint_heldout_best.pt` selected step 1 by loss
  and produced zero heldout mesh chunks with the required gate.
- `checkpoint_last.pt` produced heldout mesh chunks, but `mapped_pixel_ratio`
  was `0.999262` on heldout and `0.999353` on the 120-frame long run.
- Heldout teacher-cache final pose was `22.526x` the no-motion baseline, so the
  current student is a diagnostic/toy baseline.

Required work:

- Improve checkpoint selection so validation quality cannot choose an untrained
  step-1 checkpoint when heldout depth/pose gates are bad.
- Replace or substantially harden `SMGTTiny` pose/depth/confidence learning
  before object/dynamic fusion. Keep changes scoped to the student core,
  training losses, confidence calibration, and mapping gates.
- Add calibration diagnostics: confidence/sigma should correlate with heldout
  error and reject a meaningful fraction of high-error pixels.
- Train/evaluate on the existing strict split first. Add a second sequence/cache
  only if available locally or generated through the existing teacher bridge.
- Run heldout RGB-only mapping without VGGT at student inference using the
  validation-selected checkpoint, not only `checkpoint_last.pt`.

Acceptance:

- Heldout teacher-cache depth AbsRel `<= 0.20` or at least 25% better than the
  constant-depth baseline.
- Heldout teacher-cache pose center error beats the no-motion baseline by at
  least 15%.
- Validation-selected checkpoint produces nonzero heldout mesh chunks with
  `confidence_sigma` gating.
- `mapped_pixel_ratio` is materially below all-positive and the report explains
  the chosen threshold.
- Long-run replay has no explosive active-block growth, no nonfinite outputs,
  and no near-1.0 mapped-pixel ratio under the accepted gate.
- Full verification passes: format, lint, typecheck, full unittest discovery,
  focused tests, `git diff --check`, and `make smoke`.
