# Codex Prompt - Atlas3R Phase 5B: External Teacher Forge Adapters

Phase 5B - External Teacher Forge Adapters: dependency-isolated Depth
Pro/VGGT/LingBot-Map output ingestion into the same clip-cache/teacher-signal
format. Start with output-ingestion contracts and one local-folder adapter; do
not vendor model repos or weights.

Required boundaries:

- Do not work on `main`.
- Do not vendor external model repos or weights.
- Do not commit generated data, checkpoints, NPZ files, previews, or runs.
- Keep optional model dependencies isolated behind adapters or local-folder
  ingestion paths with clear dependency errors.
- Preserve truth flags: no benchmark accuracy, millimeter, realtime, or mapping
  readiness claims without a named evaluation report.

Starting context:

- Reload `README.md`, `PLANS.md`, `docs/08_API_CONTRACTS.md`, and
  `docs/status/*`.
- Use the Phase 5A clip cache and temporal training contracts as the shared
  ingestion target.
- Begin with contracts/tests for output ingestion before adding any external
  model runner.
