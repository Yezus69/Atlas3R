# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Current branch: `codex/core-smgt-tiny-generalization-gauntlet`.
- Core Phase A2 is implemented as a falsification gate for the first learned
  diagnostic `SMGTTiny` student. It adds confidence/sigma/dynamic mapping gates,
  deterministic train/val/heldout teacher-cache split manifests, stable training
  metric windows, heldout RGB-only mapping diagnostics, and a 120-frame long-run
  replay.
- `python -m atlas3r train smgt-tiny` can write heldout metrics,
  `smgt_tiny_split_manifest.json`, `split_report.md`, and
  `checkpoint_heldout_best.pt`.
- `python -m atlas3r inspect smgt-tiny-split` reports deterministic non-overlap
  before training.
- `python -m atlas3r runtime map-rgb-student --rgb-only` now gates mapper pixels
  by depth range, confidence, optional sigma, and dynamic probability. Summaries
  report raw, confidence-gated, sigma-gated, dynamic-rejected, and mapped pixel
  ratios; `all_positive` remains unsafe diagnostic-only.
- Truth flags keep `teacher_geometry_used=false` during student inference,
  measured depth/pose false for mapping, `metric_scale_source` as unverified
  student RGB prior, and no final SMGT, RGB-only readiness, realtime,
  object-aware fusion, hidden geometry, accuracy, or millimeter claim.

## Latest Verified Test State

- Final verification passed on 2026-06-05:
  `python -m ruff format src tests`, `python -m ruff format --check src tests`,
  `python -m ruff check src tests`, `python -m mypy src`,
  `python -m unittest discover -s tests -p "test_*.py"`, focused A2 unit tests,
  and `git diff --check`.
- `make smoke` could not run because `make` is not installed on this Windows
  host (`spawnSync make ENOENT`). The five Python commands from the Makefile
  `smoke` target were run directly and all passed.

## A2 Real-Data Evidence

- Teacher cache:
  `runs/phase6h_rgb_teacher_stitched_freiburg1_xyz_val/teacher_temporal_cache`
  with 29 validated pseudo clips.
- Strict split: train frames `676-747`, val `748-771`, heldout `772-795`;
  boundary clips `17` and `23` omitted; no frame-ID overlap.
- Training command: `python -m atlas3r train smgt-tiny --teacher-cache
  runs/phase6h_rgb_teacher_stitched_freiburg1_xyz_val/teacher_temporal_cache
  --output runs/core_smgt_tiny_a2_weighted_split_freiburg1_xyz_val --steps 5000
  --batch-size 4 --device cuda:0 --amp --val-split 0.2 --heldout-split 0.2
  --save-every 500`.
- Final heldout teacher-cache depth AbsRel was `0.188903`, but depth was still
  `1.883660x` the constant-depth baseline and pose was `22.526175x` the
  no-motion baseline.
- The validation-selected `checkpoint_best.pt` stayed at step `1` and produced
  zero heldout mesh chunks with the required `confidence_sigma` gate.
- The trained `checkpoint_last.pt` produced 45 heldout mesh chunks with 18,972
  vertices and 9,486 triangles, but `mapped_pixel_ratio` was `0.999262`.
- The unsafe all-positive comparison produced the same heldout mesh counts and
  `mapped_pixel_ratio=1.0`; the stricter gate filtered only 340 of 460,800
  pixels.
- The 120-frame long run produced 113 mesh chunks, 75,580 vertices, 37,790
  triangles, active blocks/voxels `141 / 24,519`, and `mapped_pixel_ratio`
  `0.999353`. Sampled peak process RSS was `1,164.06 MiB`; sampled peak GPU0
  memory was `1,818 MiB`.
- Full evidence: `docs/status/core_smgt_tiny_a2_generalization_report.md`.

## Current Known Gaps

- The current `SMGTTiny` is a diagnostic/toy baseline, not a validated
  foundation for object/dynamic fusion.
- Training used VGGT teacher pseudo labels, not measured geometry labels.
- Metric scale is an unverified RGB prior learned from pseudo labels.
- Confidence/sigma calibration is not selective enough; mapping remains near
  all-positive.
- No object-aware fusion, dynamic filtering, loop closure, global optimization,
  realtime proof, benchmark accuracy report, or millimeter claim exists.
