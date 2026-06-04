Phase 5H.1 - Configure And Run Real VGGT Teacher Evidence.

Goal: unblock Phase 5H by installing or pointing Atlas3R at a real VGGT
checkout and checkpoint outside this repo, then generate and evaluate external
VGGT pose/depth/pointmap teacher signals before any student training.

Start from branch `codex/phase5h-vggt-pose-pointmap-teacher`. Reload
`README.md`, `PLANS.md`, `docs/08_API_CONTRACTS.md`,
`docs/status/progress.md`, `docs/status/decisions.md`,
`docs/status/active_task.md`, and
`docs/status/phase5h_vggt_pose_pointmap_teacher_report.md`.

Required setup:

- Set `ATLAS3R_VGGT_REPO` to a local VGGT checkout or install an importable
  `vggt` package in the active environment.
- Set `ATLAS3R_VGGT_CHECKPOINT` or pass `--checkpoint` if the local VGGT API
  needs explicit weights.
- Keep VGGT code, weights, checkpoints, generated teacher caches, and run
  artifacts outside git-tracked Atlas3R files.

First command to try:

```bash
python -m atlas3r teachers run-vggt \
  --clip-cache data/tum_rgbd/freiburg1_xyz_phase5g1_clip_cache_val \
  --output runs/phase5h_vggt_freiburg1_xyz_val \
  --device cuda \
  --max-clips 8 \
  --align-to-source-pose diagnostic_sim3
```

Then inspect the generated `summary.json`, `per_clip_metrics.jsonl`, and
`report.md`. Repeat on at least one additional Phase 5G.1 validation sequence if
the first run succeeds.

Training gate:

- Train the existing temporal student only if the real VGGT teacher improves
  depth RMSE, relative pose/RPE, or pointmap quality against measured TUM data
  compared with the Phase 5G.1 student evidence.
- If VGGT does not pass a gate, do not train. Update the report and try a
  different external/local measurement teacher such as LingBot-Map or
  Anchor-style local geometry.
