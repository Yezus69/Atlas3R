# Phase 6A Product-Slice Mapper Recording Mesh Report

Branch: `codex/phase6a-product-slice-mapper-recording-mesh`

Base branch: `codex/phase5h-vggt-pose-pointmap-teacher`

## Result

Phase 6A added a measured, product-slice mapping path:

```text
Atlas3R recording
  -> validated RGB + K + measured depth + measured pose frames
  -> chronological DepthObservation stream
  -> CPU TSDF fusion
  -> PLY point cloud export
  -> optional real marching-cubes mesh export when scikit-image is installed
  -> latency, memory, quality, event, summary, and preview reports
```

No model, teacher wrapper, training run, generated data, checkpoints, or mesh
artifacts were committed. Runtime outputs are diagnostic only and make no
realtime, mapping-readiness, benchmark, or millimeter-accuracy claim.

## Code Changed

- Added `src/atlas3r/recording/` with the `atlas3r_recording` manifest/frame
  schema, safe path validation, TUM import, clip-cache import, and
  recording-to-`DepthObservation` conversion.
- Added `atlas3r recording validate`, `atlas3r recording from-tum`, and
  `atlas3r recording from-clip-cache`.
- Added `atlas3r runtime fuse-recording` for measured recording fusion through
  the existing CPU TSDF path.
- Added optional `src/atlas3r/mapping/mesh_extraction.py` using
  `scikit-image` marching cubes when installed, with a clear `mesh_status.json`
  fallback when absent.
- Added `mesh = ["scikit-image"]` as an optional extra; base dependencies stay
  unchanged.
- Added focused tests for schema validation, path traversal rejection,
  TUM/clip-cache import, runtime fusion, PLY output, missing mesh dependency,
  CLI help, and status handoff.
- Updated `docs/08_API_CONTRACTS.md`, `docs/status/decisions.md`, compact
  progress, and next-task handoff.

## Real Commands Run

The goal-file manifest name was not present, so the available Phase 5G.1 block
manifest was used.

```bash
python -m atlas3r recording from-tum \
  --manifest data/tum_rgbd/freiburg1_xyz_phase5g1_manifest_block.json \
  --output runs/phase6a_recording_freiburg1_xyz_val/recording \
  --split val \
  --max-frames 120 \
  --width 160 \
  --height 120

python -m atlas3r recording validate \
  --input runs/phase6a_recording_freiburg1_xyz_val/recording

python -m atlas3r runtime fuse-recording \
  --recording runs/phase6a_recording_freiburg1_xyz_val/recording \
  --output runs/phase6a_fuse_recording_freiburg1_xyz_val \
  --pose-source recording \
  --depth-source recording \
  --max-frames 120 \
  --keyframe-stride 1 \
  --voxel-size-m 0.05 \
  --truncation-voxels 3.0 \
  --export-point-cloud \
  --export-mesh auto
```

## Real Run Evidence

- Source dataset/sequence: TUM RGB-D `freiburg1_xyz`.
- Split: `val`.
- Recording shape: 160x120.
- Frames imported and validated: 120.
- Frames fused: 120, frame IDs 676 through 795.
- Surface point count: 3,136.
- Point cloud: `surface_points.ply` written under the ignored run directory.
- Triangle mesh: not exported because optional `scikit-image` is not installed.
  `mesh_status.json` records `mesh_exported=false` and the install hint.
- Valid measured depth pixels: 1,714,731.
- Valid measured depth ratio: 0.7442408854166667.
- Valid measured depth range: min 0.6633999943733215 m, mean
  1.0846057994254343 m, max 4.11460018157959 m.

Latency report, wall-clock diagnostic milliseconds:

| Segment | p50 | p95 | max |
| --- | ---: | ---: | ---: |
| recording_observation_load | 1891.5437 | 1891.5437 | 1891.5437 |
| cpu_tsdf_mapping | 3973.6902 | 3973.6902 | 3973.6902 |
| geometry_export | 26.2614 | 26.2614 | 26.2614 |
| total_pipeline | 5892.6372 | 5892.6372 | 5892.6372 |

Memory counters are deterministic NumPy array byte counts, not process RSS:

- observation arrays: 27,648,000 bytes.
- TSDF arrays: 2,457,216 bytes.
- surface arrays: 62,720 bytes.

## Known Limitations

- CPU TSDF is diagnostic and not the final realtime GPU mapper.
- `quality_report.json` summarizes measured input coverage; it is not a
  prediction benchmark or accuracy report.
- Mesh extraction needs optional `scikit-image`; without it the runtime exports
  only the point cloud and mesh status.
- TUM import references external source files through a declared root; deleting
  the source data invalidates the recording.
- No live/video capture, calibration capture, object-aware fusion, GPU bounded
  mapper, or glTF/game-engine mesh export exists yet.

## Exact Next Bottleneck

Phase 6B should add live/video input and calibration capture so real apartment
captures can produce validated `atlas3r_recording` folders. The blocker is now
input capture/calibration, not another model wrapper.

## Verification

Commands run:

```bash
python -m ruff format src tests
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m unittest discover -s tests -p "test_*.py"
git diff --check
where.exe make
```

Results:

- `python -m ruff format src tests`: passed; 163 files left unchanged.
- `python -m ruff format --check src tests`: passed; 163 files already
  formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 122 source files.
- `python -m unittest discover -s tests -p "test_*.py"`: passed 208 tests; the
  pre-existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git warned changed LF files will convert to CRLF.
- `where.exe make`: no `make` found, so `make test`, `make lint`, and
  `make typecheck` were unavailable in this Windows shell.
