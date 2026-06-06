# Architecture Decisions

This is a compact ADR index for the reset foundation.

| ID | Area | Decision | Still active? |
| --- | --- | --- | --- |
| D-0001 | Direction | Offline World Builder is the immediate foundation; realtime checkpoint work is downstream of offline labels. | Yes |
| D-0002 | Truth boundary | Measured GT, anchored labels, synthetic GT, CAD-aligned labels, teacher pseudo labels, and unanchored MP4 pseudo labels remain separate. | Yes |
| D-0003 | Import safety | `import atlas3r` must not import Torch, OpenCV, teacher models, Open3D, or external reconstruction packages. | Yes |
| D-0004 | Teacher role | Depth Pro, VGGT, MapAnything, LingBot-Map, SAM/DINO, CoTracker, and COLMAP/GLOMAP are witnesses, not truth. | Yes |
| D-0005 | Coordinates | Camera frame is x-right, y-down, z-forward; transforms use `T_A_B` names and meters by default. | Yes |
| D-0006 | Artifacts | Map artifacts must carry source frame IDs, observed coverage, voxel size when relevant, coordinate frame, scale source, and uncertainty. | Yes |
| D-0007 | CLI | Active CLI is limited to `offline inspect-video`, `teachers list`, and `smoke contracts`. | Yes |
| D-0008 | Stale code | Old SMGT, training, measured replay, runtime student mapping, and phase reports are deleted rather than carried as legacy code. | Yes |
| D-0009 | Offline milestones | Future work must advance through `offline build-world` vertical slices, not isolated single-stage modules. | Yes |
| D-0010 | Debug geometry | `debug_flat_depth` may create preview geometry only under explicit debug modes and must not claim measured geometry or training quality. | Yes |
| D-0011 | VGGT witness | VGGT runtime/replay is a vertical teacher-proposal path through `offline build-world`; its unanchored geometry is not measured and not training-quality. | Yes |
| D-0012 | Window stitching | V0.6 uses only minimal overlap Sim3 stitching and creates pseudo-submaps on rejected overlaps rather than silently fusing windows. | Yes |
| D-0013 | Depth Pro disagreement | Depth Pro is a second depth/intrinsics witness; V0.7 records VGGT-vs-Depth-Pro disagreement and diagnostic consensus without claiming optimization, physical accuracy, or training quality. | Yes |
| D-0014 | Fused map output | V0.8 exports teacher-pseudo fused points, sparse occupancy, observed voxel mesh, and camera trajectory only when global pose exists; Depth Pro alone cannot create a global map. | Yes |
| D-0015 | Map optimizer | V0.9 optimizes only diagnostic teacher consistency using Depth Pro scale/bias, optional focal scale, confidence updates, and projection residuals; optimized maps remain teacher-pseudo and not measured. | Yes |
| D-0016 | Soft-metric best map | V1.0 no-anchor room maps select an inspectable `world_map_best/` from optimized/raw/fallback teacher-consensus artifacts and record `soft_metric_unanchored` scale confidence without physical or training-quality claims. | Yes |
| D-0017 | Classical witness | V1.1 treats COLMAP/GLOMAP as unanchored SfM proposals behind external-process/replay boundaries; missing executables and failed stages are first-class reports, not hidden behind VGGT/Depth-Pro success. | Yes |
