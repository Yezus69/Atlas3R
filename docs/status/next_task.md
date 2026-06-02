# Codex Prompt - Atlas3R Phase 5A: Real Multi-View Clip Forge + Temporal TUM

Start on a non-main branch after Phase 4D is committed. Build the reusable TUM
RGB-D multi-view clip cache and tiny temporal geometry training path. Do not add
external model repositories. Keep generated `data/`, `runs/`, checkpoints, NPZs,
previews, and cache payloads ignored.

Required endpoints:

- `atlas3r forge tum-rgbd-clips`
- `atlas3r train tum-rgbd-temporal`

Required code:

- `src/atlas3r/forge/clip_cache.py`
- `src/atlas3r/forge/tum_rgbd_clips.py`
- `src/atlas3r/training/tum_clip_dataset.py`
- `src/atlas3r/training/tiny_temporal_geometry_model.py`
- `src/atlas3r/training/temporal_losses.py`

Verify with ruff format/check, mypy, unittest, and `git diff --check`. If TUM
data and CUDA are available, forge train/val clip caches and run temporal-v0.
Write `docs/status/phase5a_real_multiview_forge_temporal_report.md`, update this
file to Phase 5B, and commit code/docs only.
