Core Phase A4 - Broaden Student Validation And First-Class Profiling

Goal: harden SMGT-small-v2 across more measured heldout evidence before object/dynamic fusion.
Do not add SAM, object-aware sparse fusion, or dynamic-scene fusion until this
broader validation passes.

Start from branch `codex/core-smgt-small-v2-measured-pseudo`. Reload
`README.md`, `PLANS.md`, `docs/01_SYSTEM_ARCHITECTURE.md`,
`docs/08_API_CONTRACTS.md`, `docs/09_EVALUATION.md`,
`docs/status/progress.md`, `docs/status/decisions.md`,
`docs/status/active_task.md`, and
`docs/status/core_smgt_small_v2_measured_pseudo_report.md`.

Context:

- Core A3 added measured temporal caches, SMGT-small-v2 measured/pseudo
  training, validation-quality checkpoint selection, confidence/sigma gate
  calibration, and `runtime map-rgb-student-v2`.
- A3 selected `checkpoint_best.pt` at step 1500 from
  `runs/core_smgt_small_v2_train_freiburg1_freiburg2`.
- Heldout Freiburg validation passed local diagnostic gates: depth beat the
  constant-depth baseline, pose beat no-motion, calibrated mapped ratio was
  `0.311182`, and 32 observed mesh chunks were emitted.
- The result is still diagnostic: no final SMGT, no object-aware fusion, no
  realtime proof, no benchmark accuracy report, no millimeter claim, and
  `metric_scale_source=student_rgb_prior_unverified`.

Required work:

- Add first-class runtime memory/profile instrumentation to
  `runtime map-rgb-student-v2` summaries instead of relying on external samples.
- Run at least one additional measured heldout sequence/cache if locally
  available, or prepare one through existing TUM recording/cache commands.
- Verify calibrated confidence/sigma gates remain selective across sequences and
  mapped pixels keep lower error than rejected pixels.
- Preserve the A3 Freiburg gates as regression checks.
- Keep measured depth/pose as training/eval-only; runtime mapping must remain
  RGB-only and teacher-free.

Acceptance:

- Existing A3 Freiburg heldout metrics do not regress under the new code.
- Additional heldout sequence produces nonzero mesh chunks with
  `confidence_sigma` gating or the report clearly marks the sequence-specific
  failure.
- Mapped-pixel ratio remains materially below all-positive and within the
  calibrated target unless explicitly justified.
- Runtime summary records latency percentiles and memory counters without
  external sampling.
- Full verification passes: format, lint, typecheck, full unittest discovery,
  focused tests, `git diff --check`, and `make smoke` or documented direct
  equivalents on Windows.
