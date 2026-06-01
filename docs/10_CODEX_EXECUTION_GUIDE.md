# 10 — Codex Execution Guide

Use these prompts/tasks with Codex. Each task should be a small, testable vertical slice.

## General Codex prompt prefix

```text
Read AGENTS.md, README.md, PLANS.md, and only the docs relevant to this task. Do not paste huge code in chat. Modify files directly. Add tests. Run the relevant make command. Summarize changed files, tests run, and known gaps.
```

## Task 0 — Initialize repository

```text
Create the Atlas3R repository skeleton exactly as README.md specifies. Add pyproject.toml, Makefile, package dirs, test dirs, docs/status files, and a minimal CLI entry point `atlas3r --help`. Add placeholder commands for `make format`, `make lint`, `make typecheck`, `make test`, `make smoke`. Do not implement neural models yet.
```

Expected output:

- importable package;
- CLI help works;
- test runner passes with placeholder tests.

## Task 1 — Core data contracts

```text
Implement the public data contracts from docs/08_API_CONTRACTS.md using dataclasses and numpy typing. Add validation helpers for transforms, intrinsics, image shapes, and meters units. Add unit tests for valid and invalid contracts.
```

Expected tests:

- valid `T_world_camera` accepted;
- non-4×4 transform rejected;
- invalid intrinsics rejected;
- mesh chunk metadata required.

## Task 2 — Synthetic cube-room generator

```text
Implement a deterministic synthetic cube-room dataset generator with exact camera intrinsics, poses, depth maps, object masks, and a ground-truth mesh. Add projection/unprojection tests and a smoke command that writes a tiny session folder.
```

Expected tests:

- project/unproject round trip;
- rendered depth equals analytic plane intersections;
- object masks align with generated object meshes.

## Task 3 — Teacher adapter interface

```text
Implement the GeometryTeacherAdapter protocol and TeacherPrediction schema. Add stub adapters for LingBot-Map, MapAnything, VGGT, Depth Pro, and SAM3 that raise clear dependency errors if packages/checkpoints are unavailable. Add one FakeTeacherAdapter for tests.
```

Expected tests:

- fake adapter returns valid outputs;
- missing dependency errors include install instructions;
- teacher cache save/load preserves metadata.

## Task 4 — TSDF reference integrator

```text
Implement a CPU reference TSDF integrator for correctness using the contracts. It should integrate depth+pose into voxel blocks and extract a mesh with marching cubes or a simple block surface method. Prioritize correctness over speed. Add synthetic cube-room tests.
```

Expected tests:

- planar wall surface within <0.5 voxel mean error;
- object IDs carried into mesh faces;
- low-confidence pixels are skipped.

## Task 5 — Runtime scheduler skeleton

```text
Implement an async runtime scheduler with fake model predictions. It should ingest video frames or synthetic frames, emit PoseUpdate and MeshChunk events, and record latency metrics. No real neural model yet.
```

Expected tests:

- event order stable;
- dropped/late mapper frames do not block pose stream;
- runtime profile JSON produced.

## Task 6 — SMGT model skeleton

```text
Implement the Streaming Metric Geometry Transformer skeleton with a tiny encoder and heads. Focus on tensor shapes, memory state, anchor/local/trajectory context APIs, and exportable outputs. Add shape tests and synthetic overfit script.
```

Expected tests:

- forward pass on B=1,T=8,H=128,W=160;
- state update bounded by configured memory size;
- heads output correct shapes and finite values.

## Task 7 — Loss functions

```text
Implement the losses from docs/05_LOSSES.md with masks and robust reductions. Add tests with tiny tensors where expected values can be computed manually. Do not connect full training yet.
```

Expected tests:

- depth log loss;
- pointmap metric loss;
- SE(3) pose loss identity case;
- uncertainty NLL penalizes overconfident errors;
- object dice/BCE loss.

## Task 8 — Training single-batch loop

```text
Create a trainer that loads synthetic data, runs SMGT, computes losses, and performs one optimizer step. Add a 100-step overfit debug command for synthetic cube-room. Record loss curves.
```

Expected tests:

- one train step runs on CPU for tiny model;
- overfit loss decreases on GPU if available.

## Task 9 — Real teacher baseline

```text
Pick one available teacher model first, preferably LingBot-Map if installed. Implement a real adapter without vendoring the repo. Process a short video into TeacherPrediction, feed it into TSDF, and export GLB/PLY. Keep failures explicit and recoverable.
```

Expected output:

- sample session folder;
- mesh export;
- runtime/quality metadata.

## Task 10 — Performance profile

```text
Add `atlas3r profile` to measure per-stage latency, GPU memory, and FPS for fake, teacher, and student pipelines. Reports must include hardware and model checkpoint hash.
```

Expected output:

- JSON and Markdown profile report;
- p50/p90/p99 latency.

