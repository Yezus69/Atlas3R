# API Contracts

This file is a concise contract index for the pivot. It documents intended
interfaces only; no runtime implementation exists on this branch yet.

## Coordinate And Unit Rules

- Units are meters unless a field explicitly states otherwise.
- `T_world_camera` is a 4x4 camera-to-world transform.
- Camera rays are unit vectors in camera coordinates.
- Radial depth is distance along a ray.
- Unknown, free, occupied, dynamic, and predicted states must remain distinct.

## VideoInput

Purpose: identify a source RGB video or decoded frame sequence.

Required fields:

- `source_uri`
- `frame_count`
- `fps`
- `width_px`
- `height_px`
- `timestamp_s[]`
- `metadata`

Failure modes:

- missing source;
- unsupported codec;
- variable frame timing not represented;
- severe corruption.

## ReconstructabilityReport

Purpose: decide whether a video should enter reconstruction.

Required fields:

- `accepted_for_reconstruction`
- `rejection_reasons[]`
- `parallax_score`
- `blur_score`
- `dynamic_foreground_ratio`
- `zoom_or_stabilization_score`
- `static_structure_score`
- `confidence`

This report does not decide metric acceptance.

## FrameRayPacket

Purpose: canonical per-frame geometry from a backbone or optimizer.

Required fields:

- `frame_id`
- `T_world_camera`
- `rays_camera[H,W,3]`
- `radial_depth_m[H,W]`
- `confidence[H,W]`
- `camera_model`
- `source`
- `uncertainty`

Optional fields:

- `intrinsics`
- `rolling_shutter_model`
- `depth_residual_field`

## VideoGeometryBackbone

Purpose: normalize external geometry engines.

Conceptual interface:

```text
predict(video_or_keyframes) -> GeometryBackbonePrediction
```

Output contract:

- frame ray packets;
- camera confidence;
- depth confidence;
- dependency and artifact provenance;
- clear unavailable status when external dependencies are missing.

Initial planned backbones:

- default: ViPE with DA3;
- fallback: MegaSaM for selected hard videos.

## MaskTrackSet

Purpose: represent SAM2 or Grounded-SAM2 mask groups.

Required fields:

- `track_id`
- `frame_ids[]`
- `mask_rle_or_bitmap`
- `mask_confidence`
- `prompt_or_source`
- `semantic_label_optional`

Mask grouping does not itself mark static or dynamic.

## StaticDynamicState

Purpose: record geometry-led static/dynamic inference.

Required fields:

- `frame_id`
- `static_probability[H,W]`
- `dynamic_probability[H,W]`
- `unknown_probability[H,W]`
- `mask_track_decisions[]`
- `residual_summary`

## ScalePosterior

Purpose: make metric scale explicit.

Required fields:

- `scale_mean`
- `scale_std`
- `relative_scale_uncertainty`
- `scale_sources[]`
- `anchor_residuals[]`
- `metric_acceptance_status`

Suggested statuses:

- `measured_metric`
- `metric_pseudo_label`
- `non_metric_pseudo_label`
- `rejected`

## OptimizedSceneState

Purpose: hold refined geometry before mapping.

Required fields:

- `frame_ray_packets[]`
- `scale_posterior`
- `static_dynamic_state[]`
- `visibility_graph`
- `optimizer_trace`
- `validation_inputs`

## MeshChunkMetadata

Purpose: keep geometry provenance attached to mesh outputs.

Required fields:

- `chunk_id`
- `source_frame_ids[]`
- `observed_coverage_estimate`
- `voxel_size_m`
- `coordinate_frame`
- `metric_scale_source`
- `mean_uncertainty_m`
- `p50_uncertainty_m`
- `p95_uncertainty_m`
- `observed_only`
- `predicted_completion`

## OccupancyGrid2D

Purpose: floor-aligned robot grid for downstream training or planning.

Required fields:

- `grid_frame`
- `resolution_m`
- `origin_world`
- `P_free[x,y]`
- `P_occupied_static[x,y]`
- `P_dynamic[x,y]`
- `P_unknown[x,y]`
- `height_min_m[x,y]`
- `height_max_m[x,y]`
- `scale_uncertainty`
- `map_confidence`

Do not collapse this into a binary occupied/free grid too early.

## ValidationReport

Purpose: decide whether outputs are accepted for metric training.

Required fields:

- `held_out_render_error`
- `free_space_contradiction_rate`
- `scale_posterior`
- `floor_wall_consistency`
- `dynamic_leakage_score`
- `accepted_for_metric_training`
- `rejection_reasons[]`

Metric acceptance requires this report plus a tight scale posterior.
