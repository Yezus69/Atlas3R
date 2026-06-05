# Offline V0.5 Parallel Tracer Report

Branch: `codex/offline-world-builder-v0-parallel-tracer`
Commit: `pending-final-commit`

## Pipeline Artifacts

`offline build-world` writes:

```text
run_manifest.json
frames/frame_index.jsonl
keyframes/keyframes.json
teachers/teacher_status.json
proposals/proposal_manifest.json
world/world_state.json
world/camera_ledger.json
world/scale_ledger.json
geometry/geometry_preview.npz
geometry/geometry_preview.ply when points exist and --write-ply is set
objects/object_ledger.json
diagnostics/render_repair_diagnostics.json
diagnostics/failure_points.json
quality_report.json
quality_report.md
training_cache/training_cache_manifest.json
```

## Modules Run

Run orchestrator, frame cache, keyframe selector, teacher witness layer,
proposal cache, camera/scale ledger, consensus world state, geometry preview,
object ledger, render/repair diagnostics, training-cache manifest, and quality
report all run from one command.

## Evidence A

Command:

```bash
python -m atlas3r offline build-world --input build/offline_v05_tiny_ppm_input --output runs/offline_v05_tiny_ppm --max-frames 12 --keyframe-stride 2 --debug-geometry-mode flat-depth --write-ply
```

Result: passed with exit code 0. Geometry preview wrote 288 points and
`geometry/geometry_preview.ply`. Quality and training manifests mark the output
as debug synthetic, not measured, not physically accurate, and not
training-quality.

## Evidence B

Local input found:
`data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb`.

Command:

```bash
python -m atlas3r offline build-world --input data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb --output runs/offline_v05_real_input --max-frames 120 --keyframe-stride 5 --keyframe-max-count 32 --debug-geometry-mode none --write-ply
```

Result: passed with exit code 0 and wrote the full artifact tree. Geometry
points: 0 because the real input is PNG and V0.5 only decodes PPM
dependency-free. The report records missing image decoder, missing geometry,
missing object witness, missing render geometry, and unavailable teachers.

## Teacher Availability

Unavailable: Depth Pro, VGGT, MapAnything, LingBot-Map, SAM/DINO, CoTracker,
and COLMAP/GLOMAP. Each has an install hint and failure point. The only geometry
producer in Evidence A is explicit `debug_flat_depth`.

## Failure Points

Evidence A found 8 expected failure points: seven unavailable real witnesses
and missing SAM/DINO object proposals.

Evidence B found 12 expected failure points: unsupported PNG folder for the
dependency-free frame cache, no keyframes, seven unavailable real witnesses,
missing geometry inputs, missing SAM/DINO object proposals, and missing
geometry for render diagnostics.

## Accuracy And Training Boundary

No output is physically accurate yet because there is no measured scale anchor,
calibration capture, optimized consensus, or named evaluation report. Debug
flat-depth geometry is inspectable only. The training cache remains unusable
for real model training because no real optimized labels exist.

## Next Bottleneck

Offline V0.6 should wire the first real geometry witness, preferably VGGT or
Depth Pro, through the existing proposal cache and tracer so reports can move
from debug/empty geometry to teacher-proposed geometry.

## Verification

Passed:

- `python -m ruff format src tests`
- `python -m ruff format --check src tests`
- `python -m ruff check src tests`
- `python -m mypy src`
- `python -m unittest discover -s tests -p "test_*.py"`: 27 tests
- `git diff --check`; Git emitted Windows CRLF replacement warnings only
- `python -m atlas3r --help`
- `python -m atlas3r offline --help`
- `python -m atlas3r offline build-world --help`
- `python -m atlas3r teachers list`
- `python -m atlas3r smoke contracts`

Unavailable:

- `make test`, `make lint`, `make typecheck`, and `make smoke` because `make`
  is not installed on this Windows host (`make` is not recognized).
