# 04 - API Contracts

This is the concise contract index for Offline World Builder V0. Units are
meters unless a field says otherwise. Transform names use `T_A_B`, mapping
points from frame `B` into frame `A`.

## Coordinate Convention

- Camera frame: `x` right, `y` down, `z` forward.
- World frame: initialized from an anchor or from the first consensus keyframe.
- Public transforms use names such as `T_world_camera`, not `pose`.

## Essential Contracts

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
- `ObjectMaskProposal`: frame ID, teacher name, mask, confidence, label,
  optional object ID, truth boundary.
- `WorldState`: world ID, poses, cameras, teacher proposals, map artifacts,
  truth boundary, metadata.
- `MapArtifact`: artifact type, relative path, coordinate frame, source frame
  IDs, voxel size, observed coverage, mean/p95 uncertainty, truth boundary.
- `TruthBoundary`: label type, metric scale source, measured-geometry flag,
  observed/predicted flags, hidden-geometry flag, accuracy and realtime flags.

## Import Safety

`import atlas3r` must not import Torch, OpenCV, Open3D, Depth Pro, VGGT, SAM,
DINO, COLMAP, or other heavy optional dependencies. Third-party model code must
live behind dependency-safe adapters.
