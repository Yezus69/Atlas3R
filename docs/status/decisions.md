# Architecture Decisions

This file is a compact ADR index. Detailed rationale lives in git history,
tests, and the contract docs.

| ID | Area | Decision | Still active? |
| --- | --- | --- | --- |
| D-0001 | Dense matches | `DenseMatchSet` is source/target frame IDs, Nx2 source/target pixels, and N confidences in `[0, 1]`. | Yes |
| D-0002 | Synthetic sessions | Phase 0B writes `mesh_chunks/chunk_<id>_v<version>.json` as a dependency-light MeshChunk sidecar before GLB export exists. | Yes |
| D-0003 | Session inspection | `atlas3r inspect session` reads Phase 0B `.atlas3r` sidecars and writes deterministic stdlib/NumPy HTML/SVG previews. | Yes |
| D-0004 | CPU TSDF smoke | `atlas3r smoke tsdf-cube-room` writes `tsdf_grid.npz`, `surface_points.npz`, `metadata.json`, and `metrics.json`. | Yes |
| D-0005 | Teacher adapters | Teacher integrations use dependency-safe `FrameBatch`, `TeacherPrediction`, `GeometryTeacherAdapter`, capability/status metadata, and runtime dependency errors. | Yes |
| D-0006 | Teacher cache | Teacher predictions serialize to deterministic `teacher_cache/metadata.json` and `frame_summaries.jsonl`; full arrays are separate optional payloads. | Yes |
| D-0007 | Fixture teacher | `fixture-cube-room` is the only available dependency-free cache-producing adapter and is synthetic-only. | Yes |
| D-0008 | Cache arrays | `--store-arrays` opts into per-frame `arrays/frame_<frame_id:06d>.npz` payloads while summaries-only caches stay valid by default. | Yes |
| D-0009 | Cache replay | Newly written frame summaries include replay camera/pose metadata; `atlas3r smoke teacher-cache-tsdf` rejects summaries-only caches. | Yes |
| D-0010 | MeshChunk sidecar | CPU TSDF surface samples can produce an observed-only low-fidelity `mesh_chunk_sidecar.json` without GLB/PLY or marching cubes. | Yes |
| D-0011 | WorldMap sidecar | CPU TSDF outputs can produce an observed-only `world_map_sidecar.json` wrapping one validated MeshChunk and no invented objects/keyframes. | Yes |
| D-0012 | TSDF inspection | `atlas3r inspect tsdf-output` validates surface, mesh, world-map, or complete CPU TSDF output folders with deterministic JSON. | Yes |
| D-0013 | DepthObservation | Mapper fusion consumes public `atlas3r.mapping.observations.DepthObservation` instead of synthetic fixture frame types. | Yes |
| D-0014 | Mapper boundaries | Synthetic-to-observation conversion lives in `atlas3r.data.synthetic_observations`; TSDF grid helpers live in `atlas3r.mapping.tsdf_grid`. | Yes |
| D-0015 | Runtime fixture | `atlas3r smoke runtime-fixture` runs a deterministic single-threaded synthetic scheduler skeleton with event log and summary outputs. | Yes |
| D-0016 | Runtime inspection | `atlas3r inspect runtime-fixture` validates the Phase 2D output folder and nested complete TSDF inspection as a diagnostic artifact. | Yes |
| D-0017 | Student boundary | `atlas3r.models.student` provides NumPy-only `StudentClipInput`, `StudentForwardOutput`, and `ShapeOnlyStudentModel` contracts. | Yes |
| D-0018 | RGB frame source | `atlas3r.data.frame_source` provides `RGBFrameSource`, NPZ loading, and binary PPM sequence loading into existing `FramePacket` records. | Yes |
| D-0019 | Student clip bridge | `atlas3r.data.student_clip_from_frame_packets` is the public dependency-free bridge from ordered `FramePacket` sequences to `StudentClipInput`. | Yes |
| D-0020 | Teacher batch bridge | `atlas3r.data.teacher_frame_batch_from_frame_packets` and `teacher_frame_batch_from_rgb_source` bridge ordered `FramePacket` / `RGBFrameSource` inputs to existing teacher `FrameBatch`. | Yes |

Active cross-cutting constraints:

- Coordinate fields remain explicit (`T_world_camera`, `x_right_y_down_z_forward`);
  units are meters unless a field says otherwise.
- Geometry outputs carry confidence/uncertainty and must not mark hidden or
  completed geometry as measured.
- Diagnostic smoke and inspection outputs are not accuracy reports or
  performance reports.
- Third-party models remain behind adapters and must not import optional
  dependencies at module import time.
