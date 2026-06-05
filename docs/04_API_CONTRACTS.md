# 04 - API Contracts

This is the concise contract index for Offline World Builder V0.5. Units are
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
- `FrameRecord`: stable frame ID, timestamp, copied frame path, source URI,
  dimensions, guessed/known camera metadata, blur/exposure/visual-change
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
  metadata JSON, observed/predicted flags, and optional PLY point preview.
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
