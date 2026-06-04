# Progress Summary

This file is a rolling current-state summary, not an append-only transcript.
Detailed history belongs in git commits, tests, and reports.

## Current State

- Current phase: Phase 6A product-slice measured recording mapper completed.
- Branch: `codex/phase6a-product-slice-mapper-recording-mesh`.
- Latest implementation: `atlas3r_recording` validates RGB + K + optional
  measured depth + optional measured pose streams; `recording from-tum` and
  `recording from-clip-cache` import existing measured data; `runtime
  fuse-recording` streams measured recording frames through `DepthObservation`
  and CPU TSDF, then writes TSDF artifacts, `surface_points.ply`, optional real
  mesh status/export, event logs, summary, latency, memory, quality, and preview
  reports.
- Mesh extraction is optional through `scikit-image`; local run degraded
  correctly to point-cloud export with `mesh_exported=false` because
  `scikit-image` is not installed.

## Latest Verified Test State

- `python -m ruff format src tests`: passed; 163 files left unchanged.
- `python -m ruff format --check src tests`: passed; 163 files already
  formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 122 source files.
- `python -m unittest discover -s tests -p "test_*.py"`: passed 208 tests. The
  pre-existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git warned changed LF files will convert to CRLF.
- `where.exe make`: no `make` found in this Windows shell, so `make test`,
  `make lint`, and `make typecheck` were not run.

## Real-Data Evidence

- Phase 6A measured recording run:
  - Import command:
    `python -m atlas3r recording from-tum --manifest data/tum_rgbd/freiburg1_xyz_phase5g1_manifest_block.json --output runs/phase6a_recording_freiburg1_xyz_val/recording --split val --max-frames 120 --width 160 --height 120`.
  - Validation command:
    `python -m atlas3r recording validate --input runs/phase6a_recording_freiburg1_xyz_val/recording`.
  - Fusion command:
    `python -m atlas3r runtime fuse-recording --recording runs/phase6a_recording_freiburg1_xyz_val/recording --output runs/phase6a_fuse_recording_freiburg1_xyz_val --pose-source recording --depth-source recording --max-frames 120 --keyframe-stride 1 --voxel-size-m 0.05 --truncation-voxels 3.0 --export-point-cloud --export-mesh auto`.
  - Result: 120 TUM `freiburg1_xyz` validation frames fused, frame IDs
    676-795, 3,136 surface points, `surface_points.ply` exported.
  - `mesh_status.json`: `mesh_exported=false`; optional `scikit-image` missing.
  - Diagnostic latency p50/p95/max ms: observation load 1891.5437, CPU TSDF
    3973.6902, geometry export 26.2614, total pipeline 5892.6372.
  - Memory byte counters: observation arrays 27,648,000; TSDF arrays
    2,457,216; surface arrays 62,720.
  - Valid measured depth pixels: 1,714,731; valid ratio 0.7442408854166667.
  - Full evidence: `docs/status/phase6a_product_slice_mapper_recording_mesh_report.md`.
- Phase 5H real VGGT run attempt:
  `python -m atlas3r teachers run-vggt --clip-cache data/tum_rgbd/freiburg1_xyz_phase5g1_clip_cache_val --output runs/phase5h_vggt_freiburg1_xyz_val --device cuda --max-clips 1 --align-to-source-pose diagnostic_sim3`.
  It exited with code `2` because VGGT is unavailable, and no output folder was
  created.
- TUM sequences prepared for Phase 5G.1: `freiburg1_xyz`,
  `freiburg1_desk`, and `freiburg2_xyz`.
- `freiburg1_desk` and `freiburg2_xyz` downloaded successfully from the
  official TUM RGB-D URLs. `freiburg1_xyz` reused existing local data.
  `freiburg3_long_office_household` was not attempted after two additional
  successful sequences to keep the run practical.
- Phase 5G.1 cache policy: block split, validation fraction `0.15`, 160x120,
  clip length 5, stride 2, max frame gap `0.12s`, measured TUM depth/pose
  teacher caches.
- Train/val measured signals:
  - `freiburg1_xyz`: 336 train, 58 val.
  - `freiburg1_desk`: 251 train, 43 val.
  - `freiburg2_xyz`: 1,556 train, 273 val.
- Final training: 30,000 steps on CUDA with AMP, completed normally. Best
  checkpoint was step 24,500 with aggregate validation RMSE `0.265286` m, MAE
  `0.180751` m, AbsRel `0.187038`, ATE-like center mean `0.007248` m,
  relative translation mean `0.005221` m, relative rotation mean `0.256970` deg,
  RPE translation mean `0.003630` m, and RPE rotation mean `0.250807` deg.
- Best validation by sequence:
  - `freiburg1_xyz`: RMSE `0.184351` m, AbsRel `0.115135`.
  - `freiburg1_desk`: RMSE `0.521650` m, AbsRel `0.723090`.
  - `freiburg2_xyz`: RMSE `0.240855` m, AbsRel `0.117443`.
- Runtime over 60 validation frames with `checkpoint_best.pt`:
  - `freiburg1_xyz` student-odometry: depth RMSE `0.186342` m, AbsRel
    `0.086612`, ATE mean `0.137798` m, rotation mean `2.568183` deg, RPE
    translation mean `0.010458` m, RPE rotation mean `0.479756` deg.
  - `freiburg1_desk` student-odometry: depth RMSE `0.514743` m, AbsRel
    `0.719807`, ATE mean `0.425468` m, rotation mean `14.102981` deg.
  - `freiburg2_xyz` student-odometry: depth RMSE `0.281236` m, AbsRel
    `0.083914`, ATE mean `0.100351` m, rotation mean `1.429469` deg.
  - `freiburg1_xyz` oracle pose: same depth metrics as student-odometry with
    zero translation ATE, confirming depth and pose are separable there.
- Full evidence: `docs/status/phase5g1_multisequence_tum_generalization_report.md`.

## Compact Phase Ledger

- Skeleton through Phase 4: contracts, synthetic correctness, CPU TSDF
  diagnostics, dependency-safe adapters, NumPy student boundary, optional Torch
  training MVP, checkpoint TSDF bridge, and TUM RGB-D debug train/eval.
- Phase 5A: canonical multi-view TUM clip cache plus tiny temporal center-depth
  and relative-translation training path.
- Phase 5B/5B.1: stable teacher-signal cache, measured forge, local ingest,
  inspection, and deduped map replay.
- Phase 5C: dependency-safe external teacher runner boundary, Depth Pro
  bootstrap, VGGT local ingest scaffold, validated external signal caches.
- Phase 5D: teacher-signal temporal-v1 training, weighted losses, checkpoint
  export, and real measured TUM run/eval.
- Phase 5E: streaming checkpoint-to-map runtime diagnostic over real TUM val.
- Phase 5F: real Depth Pro pseudo-label cache generation, mixed measured+pseudo
  training, student export, and runtime comparison.
- Phase 5G: measured SE(3) pose supervision, 6D rotation head, SE(3) metrics,
  and diagnostic student-odometry runtime.
- Phase 5G.1: real multi-sequence TUM measured training/evaluation; training
  works, but depth generalization is uneven and student-odometry is still too
  drifty for mapping.
- Phase 5H: dependency-safe real VGGT runner/evaluator added; local execution
  blocked by missing external VGGT install/config, with no fake outputs.

## Current Known Gaps

- No realtime scheduler, live camera loop, calibration capture, bounded GPU
  TSDF, object-aware mapping, glTF/game-engine export, or benchmark accuracy
  report exists.
- Real triangle mesh extraction is implemented but requires optional
  `scikit-image`; it was not available locally for the Phase 6A run.
- `student-odometry` is not mapping-ready; the Phase 5G.1 60-frame rollout had
  large absolute drift, especially `freiburg1_desk` at `0.425468` m mean camera
  center error and `14.102981` deg mean rotation error.
- More measured TUM sequences alone did not fix generalization. The next
  blocker is stronger external pose/pointmap teachers plus model/data
  robustness before mesh/object evaluation.
- VGGT is not currently installed/configured. Set `ATLAS3R_VGGT_REPO` or install
  an importable `vggt` package and provide a checkpoint if required before
  rerunning Phase 5H teacher generation.
