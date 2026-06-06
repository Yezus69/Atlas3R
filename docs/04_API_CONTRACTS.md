# 04 - API Contracts

This is the concise contract index for Offline World Builder V0.8. Units are
meters unless a field says otherwise. Transform names use `T_A_B`, mapping
points from frame `B` into frame `A`.

## Coordinate Convention

- Camera frame: `x` right, `y` down, `z` forward.
- World frame: initialized from an anchor or from the first consensus keyframe.
- Public transforms use names such as `T_world_camera`, not `pose`.

## Existing Core Contracts

- `FramePacket`: `frame_id`, `timestamp_ns`, `rgb_u8 H,W,3`, optional
  `K_original`, `K_model`, `resize_transform`, metadata, and source URI.
- `CameraModel`: image size, `K`, distortion model/params, rolling shutter row
  time, confidence, and source.
- `PoseEstimate`: frame ID, timestamp, `T_world_camera`, optional covariance,
  confidence, tracking state, scale source, diagnostics.
- `DepthProposal`: teacher name, camera, optional pose, depth meters,
  depth sigma meters, confidence, truth boundary, source.
- `TeacherProposal`: teacher name, frame IDs, depth proposals, object mask
  proposals, truth boundary, status, metadata.
- `WorldState`: world ID, poses, cameras, teacher proposals, map artifacts,
  truth boundary, metadata.
- `MapArtifact`: artifact type, relative path, coordinate frame, source frame
  IDs, voxel size, observed coverage, mean/p95 uncertainty, truth boundary.
- `TruthBoundary`: label type, metric scale source, measured-geometry flag,
  observed/predicted flags, hidden-geometry flag, accuracy and realtime flags.

## Offline Tracer Artifact Contracts

- `RunManifest`: run ID, command args, input path, output paths, module status
  map, artifact list, failure-point reference, started/completed timestamps.
- `FrameRecord`: stable frame ID, original video frame index when available,
  timestamp, copied normalized PPM frame path, source URI, dimensions,
  guessed/known camera metadata, decoder name, blur/exposure/visual-change
  scores, and truth boundary.
- `KeyframeRecord`: frame ID, timestamp, source frame path, quality scores,
  selection rank, and reasons. It contains no pose assumption.
- `TeacherWitnessStatus`: teacher name, availability, capabilities, install
  hint, reason, proposal stream path, and optional failure point.
- `TeacherProposalCache`: manifest of normalized proposal streams, including
  depth/intrinsics/pose/mask/track availability, truth labels, uncertainty,
  coordinate convention, and source.
- `CameraScaleLedger`: camera entries, intrinsics status, scale source,
  anchoring state, physical-accuracy permission, and why any claim is blocked.
- `ConsensusWorldState`: frame/keyframe refs, proposal refs, camera/scale
  ledger refs, `pose_status`, `depth_status`, `map_status`, uncertainties, and
  unresolved fields. It must not imply optimization ran.
- `GeometryPreview`: NPZ with `points_world_m`, colors, uncertainty, frame IDs,
  source teacher IDs, pseudo-submap IDs, metadata JSON, observed/predicted
  flags, and optional PLY point preview.
- `ObjectLedger`: object-track status, object candidates, witness sources, and
  explicit unavailable state when mask/feature proposals are missing.
- `RenderRepairDiagnostics`: render/projection status, geometry count, coverage
  placeholder, mismatch placeholder, and repair hooks for pose, depth, objects,
  and scale.
- `TrainingCacheManifest`: frame/keyframe/world/geometry refs, truth boundary,
  training usability flag, and reason labels are not yet training-quality.
- `FailurePoint`: module, code, severity, status, why, missing input,
  missing dependency, future module, and artifact path.

## Import Safety

`import atlas3r` must not import Torch, OpenCV, Open3D, Depth Pro, VGGT, SAM,
DINO, COLMAP, or other heavy optional dependencies. Third-party model code must
live behind dependency-safe adapters.

## V0.6 VGGT Proposal Streams

`offline build-world --enable-vggt` runs an external VGGT runtime when
available. `--vggt-proposal-cache <path>` replays normalized streams without
importing VGGT.

Required stream files:

- `proposals/vggt_cameras.jsonl`
- `proposals/vggt_depths.npz`
- `proposals/vggt_windows.jsonl`
- `proposals/proposal_manifest.json`

Each VGGT camera proposal includes `frame_id`, `keyframe_index`, `K`,
`T_world_camera`, `camera_center_world_m`, confidence, intrinsics/pose source,
`metric_scale_source: vggt_unanchored_metric_proposal`,
`coordinate_convention`, `pseudo_submap_id`, and a truth boundary.

Each VGGT depth proposal references NPZ array keys for depth, sigma,
confidence, and valid mask. Depth is in the VGGT unanchored proposal scale.

Truth flags for VGGT streams:

```text
label_type: teacher_pseudo
measured_geometry: false
observed_only: true
predicted_completion: false
hidden_geometry_measured: false
accuracy_report: false
realtime_claim: false
usable_for_training: false
```

## V0.7 Depth Pro Proposal Streams

`offline build-world --enable-depth-pro` runs an external Depth Pro runtime when
available. `--depth-pro-proposal-cache <path>` replays normalized streams without
importing Depth Pro.

Required stream files when proposals are available:

- `proposals/depth_pro_cameras.jsonl`
- `proposals/depth_pro_depths.npz`
- `proposals/depth_pro_frames.jsonl`
- `proposals/proposal_manifest.json`

Each Depth Pro camera proposal includes `frame_id`, `keyframe_index`, optional
`K`, `focal_px`, `fx`, `fy`, `intrinsics_source: depth_pro`,
`metric_scale_source: depth_pro_metric_proposal_unanchored`,
`coordinate_convention`, and a truth boundary.

Each Depth Pro depth proposal references NPZ array keys for depth, sigma,
confidence, and valid mask. If the runtime does not emit confidence, V0.7 uses a
conservative finite-depth/smoothness-derived confidence and records
`confidence_derived: true`.

Depth Pro truth flags match VGGT teacher-pseudo flags, with
`measured_geometry: false`, `observed_only: true`, and
`usable_for_training: false`.

## V0.7 Disagreement Diagnostics

When VGGT and Depth Pro depths overlap by frame ID, V0.7 writes:

- `diagnostics/teacher_disagreement.json`
- `diagnostics/disagreement_maps.npz`
- `diagnostics/consensus_preview.npz`

`disagreement_maps.npz` contains `frame_ids`, `abs_depth_diff_m`,
`rel_depth_diff`, `valid_overlap_mask`, and `high_disagreement_mask`.

`consensus_preview.npz` contains `frame_ids`, `consensus_depth_m`,
`consensus_confidence`, and `source_mask`, where source mask values are:

```text
0 none
1 vggt_only
2 depth_pro_only
3 agree
4 disagree
```

The consensus preview is diagnostic only, not optimized, not physically
accurate, and not training-quality.

## V0.8 Fused World Map Artifacts

`offline build-world --export-world-map` writes `world_map/` artifacts from
existing proposals. VGGT pose plus diagnostic consensus depth is preferred when
both VGGT and Depth Pro exist. VGGT depth is used for VGGT-only maps. Depth Pro
without VGGT pose writes an explicit failure reason and does not create a fake
global map.

Required files for a successful inspectable map:

- `world_map/world_map_manifest.json`
- `world_map/camera_trajectory.json`
- `world_map/fused_points.npz`
- `world_map/fused_points.ply`
- `world_map/occupancy_grid.npz`
- `world_map/occupancy_grid_metadata.json`
- `world_map/observed_voxel_mesh.ply` when requested
- `world_map/map_quality.json`
- `world_map/map_quality.md`

`fused_points.npz` contains:

- `points_world_m`: `float32[N,3]`
- `colors_u8`: `uint8[N,3]`
- `confidence`: `float32[N]`
- `source_frame_ids`: `int64[N]`
- `source_keyframe_ids`: `int64[N]`
- `depth_source_id`: `int32[N]`
- `disagreement_rel`: `float32[N]`
- `point_sigma_m`: `float32[N]`

`occupancy_grid.npz` is sparse, not dense. It contains
`voxel_indices_ijk`, `occupancy_count`, `confidence_mean`, `color_mean_u8`,
`bbox_world_min_m`, and `voxel_size_m`. Unknown space is not filled and
free-space carving is not claimed.

`camera_trajectory.json` records frame ID, keyframe ID, timestamp,
`T_world_camera`, camera center, pose source, pose confidence, metric scale
source, and pseudo-submap ID for finite pose proposals.

Every fused-map artifact carries this truth boundary:

```text
label_type: teacher_pseudo_fused_map
measured_geometry: false
observed_only: true
predicted_completion: false
hidden_geometry_measured: false
metric_scale_source: unanchored_vggt_depthpro_teacher_consensus
physical_accuracy_claim: false
training_quality: false
realtime_claim: false
optimized_world_state: false
```

`map_quality.json` may set `inspectable_map_available: true` only when fused
points, sparse occupancy, requested observed mesh, and nonzero point and voxel
counts exist. This is an inspectability verdict, not an accuracy claim.
