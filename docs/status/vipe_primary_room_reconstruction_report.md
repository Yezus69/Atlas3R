# ViPE Primary Room Reconstruction Report

## Scope

Task: use external ViPE as the primary offline geometry engine for the real
`room_walk_001/frames` capture, import ViPE artifacts into Atlas3R, and compare
against the current VGGT+DepthPro `world_map_best/`.

No SAM, object fusion, student training, anchors, or physical accuracy claims
were added.

## External ViPE Runs

ViPE lives outside this repo:

```text
C:/Users/Asav/source/repos/homebrain/external/vipe
```

Smoke output:

```text
C:/Users/Asav/source/repos/homebrain/external/vipe_runs/room_walk_001_smoke60_final
```

Main output:

```text
C:/Users/Asav/source/repos/homebrain/external/vipe_runs/room_walk_001_main240
```

Main run command shape:

```text
python run.py streams=frame_dir_stream streams.base_path=.../room_walk_001/frames
streams.frame_start=0 streams.frame_end=240 streams.frame_skip=1
pipeline=no_vda pipeline.output.save_artifacts=true
pipeline.output.save_slam_map=true pipeline.slam.ba.fused=false
```

Result: exit code `0`; stdout logged `Finished processing frames`.

## Atlas3R Import

Command:

```text
python -m atlas3r offline import-vipe --vipe-output .../room_walk_001_main240
--frames .../room_walk_001/frames --output runs/room_walk_001_vipe_import
--point-stride 8 --max-points 2000000 --voxel-size-m 0.05
--previous-map runs/room_walk_001_v11_colmap_witness/world_map_best
```

The command was run with the external ViPE venv because base Atlas3R keeps
OpenEXR optional.

Output:

```text
runs/room_walk_001_vipe_import
```

Artifacts:

- `world_map_manifest.json`
- `camera_trajectory.json`
- `fused_points.npz`
- `fused_points.ply`
- `occupancy_grid.npz`
- `occupancy_grid_metadata.json`
- `observed_voxel_mesh.ply`
- `map_quality.json`
- `map_quality.md`
- `comparison_against_previous_map.json`
- `comparison_against_previous_map.md`

## Results

- ViPE depth frames: 240.
- ViPE intrinsics rows: 240.
- ViPE pose rows: 240, but all exported `T_world_camera` rows are nonfinite.
- Imported source: `vipe_slam_dense_map_fallback`.
- Points: 168,365.
- Occupied voxels: 10,628.
- Observed mesh triangles: 42,656.
- Camera trajectory poses: 0.
- Bbox: about `4.60 m x 3.78 m x 5.24 m`.

Comparison target:
`runs/room_walk_001_v11_colmap_witness/world_map_best`.

Previous V10/V11 best map:

- Points: 1,904,976.
- Occupied voxels: 2,559.
- Observed mesh triangles: 5,600.
- Camera trajectory poses: 88.
- Bbox: about `1.72 m x 0.75 m x 0.96 m`.

The ViPE import is less collapsed by bbox heuristic, but this is only an
artifact-shape diagnostic. It is not ground truth and not a physical accuracy
claim.

## Truth Boundary

```text
label_type: teacher_pseudo_vipe_near_metric
measured_geometry: false
observed_only: true
predicted_completion: false
hidden_geometry_measured: false
metric_scale_source: vipe_near_metric_teacher
scale_status: teacher_near_metric_unanchored
physical_accuracy_claim: false
training_quality: false
usable_for_training: false
```

## Known Gaps

- ViPE pose artifact is nonfinite for this run, so no Atlas3R trajectory was
  exported.
- The dense SLAM map fallback provides inspectable geometry, not camera poses.
- Scale is teacher near-metric and unanchored.
- No measured geometry, millimeter accuracy, training-quality cache, or hidden
  completion claim exists.
