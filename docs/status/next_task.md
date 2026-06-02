# Codex Prompt - Atlas3R Phase 4D: Run Real Eval, V2 Training, And Report

Run the Phase 4D TUM RGB-D checkpoint evaluator on the block-split manifest,
train the single `tiny-v2` real-depth model if CUDA/data/checkpoints are
available, evaluate v2 with `--write-tsdf`, then write the compact committed
report. Keep all generated datasets, checkpoints, NPZs, previews, and run
folders ignored.
