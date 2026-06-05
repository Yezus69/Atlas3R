# Atlas3R

Atlas3R is a research codebase for real-time RGB neural metric mapping. The
target system is:

```text
RGB/video/live camera
  -> pose stream
  -> persistent object-aware sparse TSDF/surfel map
  -> incremental game-engine mesh chunks
```

The long-term runtime target is a compact Streaming Metric Geometry Transformer
(SMGT) student for RGB pose/depth/object/uncertainty, plus a small geometric
backend for pose graph checks, sparse mapping, object fusion, and mesh export.
Large open models such as LingBot-Map, MapAnything, VGGT, Depth Pro, SAM,
DINO, and trackers are treated as offline teachers/adapters, not as the 30 FPS
runtime.

## Truth Boundary

Atlas3R must not claim millimeter accuracy, realtime readiness, or RGB-only
mapping readiness unless a named evaluation report proves it. Monocular RGB has
scale, visibility, motion, calibration, and dynamic-scene ambiguities. Hidden
or completed geometry must be marked predicted/uncertain, not measured. Every
geometry output carries confidence or uncertainty.

## Current Phase 6G State

The current implemented slice includes an offline diagnostic RGB teacher bridge,
not final RGB-only student mapping or realtime mapping:

- `atlas3r_recording` folders store RGB plus optional measured depth and
  measured `T_world_camera`.
- `python -m atlas3r runtime fuse-recording --mode incremental --backend
  cpu-persistent|cpu-rebuild|cpu-sparse` remains the measured depth+pose TSDF
  fusion path.
- `python -m atlas3r runtime live-replay-recording ...` replays a measured
  recording with deterministic simulated pacing, bounded capture/map queues,
  explicit frame/keyframe drop reasons, sparse TSDF map updates, JSONL events,
  a summary, and a Markdown report.
- With `--export-mesh-chunks`, live replay now emits observed-only triangle
  mesh chunk updates from dirty sparse TSDF blocks under `mesh_chunks/`, with
  stable `block_<x>_<y>_<z>` chunk IDs, versioned NPZ payloads, optional PLY
  payloads, a manifest, update events, conservative truth flags, and latency
  profile counters.
- `python -m atlas3r runtime map-rgb-teacher ... --teacher vggt --rgb-only`
  runs RGB frames through VGGT teacher-pseudo depth/pose/intrinsics, writes a
  pseudo recording, fuses pseudo observations through the existing sparse TSDF
  mapper, and emits observed mesh chunks through the Phase 6F writer.
- `python -m atlas3r runtime capture-adapters list` reports dependency-safe
  camera adapter status. The OpenCV adapter does not import `cv2` at module
  import time and reports an install hint when unavailable.

Phase 6G RGB teacher outputs are diagnostics. They use teacher-pseudo depth and
pose for mapping, do not use measured depth/pose for mapping in `--rgb-only`
mode, and do not claim final RGB-only student readiness, hidden geometry,
object-aware fusion, loop closure, realtime readiness, benchmark accuracy, or
millimeter accuracy.

## Useful Commands

```bash
python -m atlas3r --help
python -m atlas3r smoke synthetic-cube-room --output build/smoke/synthetic_cube_room.atlas3r
python -m atlas3r runtime capture-adapters list
python -m atlas3r runtime live-replay-recording --recording <recording> --output <run> --target-fps 30 --mapper-backend cpu-sparse
python -m atlas3r runtime live-replay-recording --recording <recording> --output <run> --target-fps 30 --mapper-backend cpu-sparse --export-mesh-chunks --mesh-format ply
python -m atlas3r runtime map-rgb-teacher --input <recording-or-rgb-folder-or-video> --output <run> --teacher vggt --device cuda:0 --max-frames 64 --frame-stride 2 --teacher-window-size 24 --teacher-window-overlap 8 --image-size 518 --voxel-size-m 0.05 --truncation-voxels 3.0 --pixel-stride 12 --export-mesh-chunks --mesh-format ply --rgb-only
python -m atlas3r runtime fuse-recording --recording <recording> --output <run> --mode incremental --backend cpu-sparse
```

Development checks:

```bash
python -m ruff format src tests
python -m ruff check src tests
python -m mypy src
python -m unittest discover -s tests -p "test_*.py"
```

## Repository Map

- `docs/`: architecture, contracts, evaluation gates, and status.
- `src/atlas3r/api/`: public typed contracts and coordinate conventions.
- `src/atlas3r/recording/`: measured recording schema/importers.
- `src/atlas3r/runtime/`: scheduler, replay, camera adapter boundary, and
  measured runtime diagnostics.
- `src/atlas3r/mapping/`: CPU dense and sparse TSDF diagnostic mappers.
- `src/atlas3r/models/adapters/`: dependency-safe teacher adapter stubs.
- `tests/`: unit, synthetic, and integration coverage.

Read `AGENTS.md`, `PLANS.md`, `docs/01_SYSTEM_ARCHITECTURE.md`,
`docs/08_API_CONTRACTS.md`, and `docs/09_EVALUATION.md` before making
architecture changes.
