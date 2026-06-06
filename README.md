# Atlas3R

Atlas3R is being reset around one foundation:

```text
MP4/RGB video -> offline optimized 3D world -> inspectable mesh/occupancy -> training cache
```

The current repository has the first connected offline tracer, two real
teacher-witness vertical slices (VGGT and Depth Pro), a dependency-safe
classical COLMAP/GLOMAP witness wrapper, the first inspectable teacher-pseudo
fused world-map artifact, the first diagnostic consistency optimizer, and an
unanchored soft-metric room-walk `world_map_best/` export. It also has a
direct external ViPE import path for observed-only room map artifacts. It is
not a working accurate mapper yet: anchored scale, object fusion, and final
mesh reconstruction remain future work.

## Boundaries

- No realtime mapping claim exists.
- No millimeter accuracy claim exists.
- No RGB-only production readiness claim exists.
- No final neural checkpoint exists.
- Teacher models are witnesses and proposal generators, not truth.
- VGGT and Depth Pro output are teacher-proposed geometry, not measured
  geometry.
- Physical accuracy requires anchors, calibration, measured depth, LiDAR/ARKit,
  or known-scale objects.
- Unanchored MP4 input produces pseudo labels only.
- Hidden or completed geometry must be marked predicted/uncertain, not measured.
- Every geometry or map artifact must carry confidence or uncertainty metadata.

## Direction

The Offline World Builder path is:

```text
MP4/RGB input
  -> frame and keyframe cache
  -> teacher proposals from Depth Pro, VGGT, MapAnything, LingBot-Map,
     SAM/DINO, CoTracker, COLMAP/GLOMAP
  -> global teacher-consensus optimizer
  -> render-and-repair consistency loop
  -> optimized camera trajectory, intrinsics, depth, static map, object tracks
  -> inspectable mesh/occupancy/training cache
```

The realtime neural checkpoint is downstream of the offline label factory, not
the next immediate foundation.

## Current CLI

```bash
python -m atlas3r --help
python -m atlas3r offline --help
python -m atlas3r offline inspect-video --input <mp4-or-image-folder> --output <run>
python -m atlas3r offline import-vipe --vipe-output <external-vipe-run> --frames <image-folder> --output <map-folder>
python -m atlas3r offline build-world --input <mp4-or-image-folder> --output <run>
python -m atlas3r offline build-world --input <images> --output <run> --enable-vggt --write-ply
python -m atlas3r offline build-world --input <images> --output <run> --vggt-proposal-cache <cache>
python -m atlas3r offline build-world --input <images> --output <run> --enable-depth-pro
python -m atlas3r offline build-world --input <images> --output <run> --vggt-proposal-cache <vggt-cache> --depth-pro-proposal-cache <depth-pro-cache>
python -m atlas3r offline build-world --input <images> --output <run> --enable-vggt --enable-depth-pro --export-world-map --map-write-occupancy --map-write-observed-mesh
python -m atlas3r offline build-world --input <images> --output <run> --enable-vggt --enable-depth-pro --export-world-map --optimize-map-consistency --export-optimized-world-map
python -m atlas3r offline build-world --input <images> --output <run> --enable-vggt --enable-depth-pro --scale-mode unanchored-soft-metric --export-world-map --optimize-map-consistency --export-optimized-world-map --export-best-world-map
python -m atlas3r offline build-world --input <images> --output <run> --enable-vggt --enable-depth-pro --enable-colmap --export-world-map --export-best-world-map
python -m atlas3r offline build-world --input <images> --output <run> --colmap-proposal-cache <classical-cache> --export-world-map --export-best-world-map
python -m atlas3r teachers list
python -m atlas3r smoke contracts
```

Removed command families include old SMGT training, student runtime mapping,
measured live replay, teacher temporal cache training, TUM training/eval, and
old phase smoke commands.

## Repository Map

- `docs/`: reset objective, architecture, truth boundary, contracts, quality.
- `src/atlas3r/contracts/`: coordinate, frame, pose, proposal, world, artifact,
  and truth-boundary contracts.
- `src/atlas3r/input/`: dependency-safe PPM/PNG/JPG/video inspection and
  decoding primitives.
- `src/atlas3r/teachers/`: dependency-safe teacher witness registry.
- `src/atlas3r/models/adapters/`: optional external model adapters.
- `src/atlas3r/offline/`: connected Offline World Builder tracer modules.
- `src/atlas3r/mapping/`: minimal NPZ/PLY artifact writers for inspection.
- `tests/`: focused unit and vertical tracer tests.

## Build-World Artifacts

`offline build-world` writes a complete artifact tree even when teachers or
decoders are unavailable:

```text
run_manifest.json
frames/frame_index.jsonl
frames/metadata_summary.json
keyframes/keyframes.json
teachers/teacher_status.json
proposals/proposal_manifest.json
world/world_state.json
world/camera_ledger.json
world/scale_ledger.json
world/scale_hypotheses.json
world/soft_metric_scale_ledger.json
geometry/geometry_preview.npz
objects/object_ledger.json
diagnostics/render_repair_diagnostics.json
diagnostics/teacher_disagreement.json
diagnostics/disagreement_maps.npz
diagnostics/consensus_preview.npz
diagnostics/failure_points.json
world_map/world_map_manifest.json
world_map/camera_trajectory.json
world_map/fused_points.npz
world_map/fused_points.ply
world_map/occupancy_grid.npz
world_map/occupancy_grid_metadata.json
world_map/observed_voxel_mesh.ply
world_map/map_quality.json
world_map/map_quality.md
optimizer/optimizer_manifest.json
optimizer/before_metrics.json
optimizer/after_metrics.json
optimizer/depth_scale_bias.jsonl
optimizer/intrinsics_adjustments.json
optimizer/projection_residuals.npz
optimizer/optimization_trace.jsonl
optimizer/optimizer_report.md
world_map_optimized/world_map_manifest.json
world_map_optimized/camera_trajectory.json
world_map_optimized/fused_points.npz
world_map_optimized/fused_points.ply
world_map_optimized/occupancy_grid.npz
world_map_optimized/occupancy_grid_metadata.json
world_map_optimized/observed_voxel_mesh.ply
world_map_optimized/map_quality.json
world_map_optimized/map_quality.md
world_map_best/world_map_manifest.json
world_map_best/camera_trajectory.json
world_map_best/fused_points.npz
world_map_best/fused_points.ply
world_map_best/occupancy_grid.npz
world_map_best/occupancy_grid_metadata.json
world_map_best/observed_voxel_mesh.ply
world_map_best/map_quality.json
world_map_best/map_quality.md
world_map_best/topdown_preview.svg
world_map_best/inspection_instructions.md
classical/classical_status.json
classical/colmap_run_manifest.json
classical/colmap_commands.jsonl
classical/colmap_sparse_summary.json
classical/colmap_cameras.jsonl
classical/colmap_images.jsonl
classical/colmap_points3d.npz
classical/colmap_sparse_points.ply
classical/trajectory_alignment.json
classical/aligned_colmap_camera_trajectory.json
classical/aligned_colmap_sparse_points.npz
classical/aligned_colmap_sparse_points.ply
diagnostics/classical_map_comparison.json
diagnostics/classical_map_comparison.md
quality_report.json
quality_report.md
room_walk_001_report.md
training_cache/training_cache_manifest.json
```

Debug flat-depth geometry is labeled `debug_synthetic`, not measured geometry,
and is not training-quality.

VGGT and Depth Pro proposal geometry, when enabled or replayed, is labeled
`teacher_pseudo`, `measured_geometry: false`, `observed_only: true`, and remains
`usable_for_training: false` until anchoring, optimization, and evaluation
exist. V0.8 can fuse VGGT poses with VGGT or diagnostic VGGT/Depth-Pro
consensus depth into `world_map/` point, occupancy, observed voxel mesh, camera
trajectory, and map-quality artifacts. V0.9 can write `world_map_optimized/`
after a diagnostic Depth Pro scale/bias consistency pass. Raw and optimized
maps are teacher-pseudo, observed-only, unanchored, not physically accurate, and
not training-quality. V1.0 adds `--scale-mode unanchored-soft-metric`,
soft-metric scale ledgers, conservative `world_map_best/` selection, and a
top-down preview. These outputs are inspectable teacher-consensus maps, not
physical ground truth. V1.1 adds a classical COLMAP/GLOMAP witness path that
can run external executables or replay sparse text models, align classical SfM
to VGGT by Sim3, compare sparse geometry against the fused maps, and optionally
write `world_map_classical_validated/` only when agreement is strong and
anti-collapse checks pass. Classical outputs are unanchored proposals, not
measured geometry.
