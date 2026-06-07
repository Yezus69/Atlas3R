# Architecture Decisions

This is a compact ADR index for the pivot baseline.

| ID | Area | Decision | Active? |
| --- | --- | --- | --- |
| D-0001 | Direction | Atlas3R pivots to a scale-aware monocular reconstruction teacher, not the previous offline-world-builder phase stack. | Yes |
| D-0002 | Truth boundary | Arbitrary unanchored RGB video must not be labeled measured metric ground truth. | Yes |
| D-0003 | Geometry backbone | ViPE with DA3 is the planned default geometry backbone; MegaSaM is a fallback for selected hard videos. | Yes |
| D-0004 | Masking | SAM2 is for mask grouping and propagation; geometry decides static, dynamic, or uncertain. | Yes |
| D-0005 | Representation | Internal geometry is ray-map first: ray, radial depth, pose, scale, confidence, and static probability. | Yes |
| D-0006 | Scale | Global scale is a variable with posterior uncertainty and explicit source metadata. | Yes |
| D-0007 | Mapping | Mapping fuses rays so free, surface, unknown, dynamic, and predicted states remain distinct. | Yes |
| D-0008 | Acceptance | Metric pseudo-label output requires validation plus a tight scale posterior; otherwise reject or mark non-metric. | Yes |
| D-0009 | Dependencies | Third-party model repos and weights stay outside this repo behind adapters and external paths. | Yes |
| D-0010 | Cleanup | This branch is docs-only after the cleanse; old code and tests were intentionally removed. | Yes |
| D-0011 | Branches | Local branches and local remote-tracking refs were pruned to one pivot branch; hosted remote branches were left untouched. | Yes |
