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
