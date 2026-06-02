# Codex Prompt - Atlas3R Phase 4D: Real Checkpoint Inference and Mapping Smoke

Load the TUM RGB-D debug checkpoint, run it on a short held-out TUM RGB frame sequence, convert predictions to DepthObservation, feed CPU TSDF, and write predicted-vs-ground-truth depth/pose/TSDF diagnostics. Keep this as a diagnostic, not an accuracy claim. Do not add new datasets, external teacher models, DDP, or mesh export.
