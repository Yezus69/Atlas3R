# Active Task - Core SMGT Small V2 Measured/Pseudo A3

Branch: `codex/core-smgt-small-v2-measured-pseudo`

## Goal

Replace the diagnostic SMGT-tiny core with SMGT-small-v2 trained from measured
and optional pseudo temporal caches, then verify whether the validation-selected
checkpoint can map heldout RGB-only frames into nonzero mesh chunks while
beating constant-depth and no-motion baselines.

## Checklist

- [x] Confirm clean worktree and switch from
  `codex/core-smgt-tiny-generalization-gauntlet`.
- [x] Read required architecture, contracts, evaluation, status, model,
  training, runtime, mapping, and test files.
- [x] Add measured temporal cache writer, validator, dataset, and inspector.
- [x] Add SMGT-small-v2 model, config, streaming state, and checkpoint helpers.
- [x] Add v2 mixed measured/pseudo dataset, losses, eval, calibration, and
  quality-based checkpoint selection.
- [x] Add `train smgt-v2`, `eval smgt-v2-calibrate-gate`, and
  `runtime map-rgb-student-v2` CLI commands without teacher imports at runtime.
- [x] Add CPU-only unit tests for measured cache, v2 model/checkpoint, losses,
  calibration, runtime mapping, and CLI help.
- [x] Build at least two measured temporal caches from real measured recordings.
- [x] Train SMGT-small-v2 long enough for meaningful curves.
- [x] Calibrate confidence/sigma gates from validation data.
- [x] Run heldout RGB-only mapping and long-run mapping with the selected
  checkpoint.
- [x] Write `docs/status/core_smgt_small_v2_measured_pseudo_report.md`.
- [x] Update README, PLANS, architecture, contracts, evaluation, progress,
  decisions, and next-task docs based on actual A3 outcome.
- [x] Run required verification commands and commit only source/docs/tests.

## Stop Conditions

- Unknown user changes appear in the worktree.
- A second measured sequence cannot be located or prepared/downloaded.
- CUDA/Torch is unavailable and CPU training is infeasible.
- Measured-cache support would require changing the phase objective.
