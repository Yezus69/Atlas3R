Phase 5E - Student checkpoint streaming runtime and CPU/GPU mapping comparison.

Goal: load a Phase 5D temporal checkpoint in the deterministic runtime scheduler,
run it over a real RGB-D/TUM clip sequence as frame stream, emit DepthObservation
per frame, fuse to CPU TSDF, and compare latency/quality diagnostics against the
teacher-signal map. Do not claim realtime or benchmark accuracy until measured.
