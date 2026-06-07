# AGENTS.md - Atlas3R Codex Operating Rules

Codex reads this file as durable repository guidance. Keep it short. Put system spec in `ARCHITECTURE.md`; put current state and milestone order in `README.md`.

## Context Load Order

Before non-trivial work:

1. Read `ARCHITECTURE.md` first to load the full system spec and contracts.
2. Read `README.md` second to find the current state and earliest incomplete milestone.
3. Re-read this file only for operating rules.
4. Inspect the relevant implementation files before editing.

If docs conflict:

```text
ARCHITECTURE.md API/contracts win over README wording.
README current state/milestone order wins over stale code comments.
AGENTS.md only controls work style.
```

## Work Selection

Each turn should choose the highest-value coherent slice inside the earliest incomplete README milestone.

Do not jump ahead to reconstruction, optimization, mapping, validation, or export until earlier data/evidence contracts exist.

A good slice produces one of:

```text
clear runtime behavior
clear data-grounded report
clear adapter boundary
clear rejection/missing-artifact status
```

A bad slice adds broad scaffolding, fake outputs, toy examples, or code that cannot be evaluated on the canonical videos.

## Data Grounding

The project is grounded in two canonical tracks:

```text
reference_metric: public indoor RGB sequence with measured metric evidence
phone_room: user phone RGB room video
```

If an asset or external artifact is missing, report `missing_asset` or `missing_external_artifact`. Do not fabricate replacement data. Do not use synthetic toy scenes as the main evidence path.

## Testing And Verification

Do not add broad unit-test bulk or synthetic scene pipelines unless explicitly requested.

Prefer data-grounded verification reports on the canonical tracks. Small invariant checks are acceptable when they prevent dangerous mistakes such as wrong coordinate frames, invalid shapes, non-unit rays, invalid probabilities, or silent metric promotion.

## Model Boundaries

Third-party model repositories and weights remain external.

Allowed:

```text
artifact adapters
manifest readers
clear unavailable/missing-artifact errors
lazy optional imports inside adapter execution paths
```

Not allowed:

```text
vendoring model repos or weights
heavy ML imports at package import time
fake model predictions
parallel model zoo behavior without explicit diagnosis purpose
```

## Truth Invariants

Never collapse these distinctions:

```text
measured_metric vs metric_pseudo_label vs non_metric_pseudo_label vs rejected
unknown vs free vs occupied_static vs movable_static vs dynamic
radial depth vs optical z-depth
T_world_camera vs T_camera_world
mask grouping vs dynamic classification
mesh quality vs robot occupancy quality
```

Metric output requires both scale evidence and validation. A nice visualization is not ground truth.

## README Discipline

`README.md` is a bounded state file, not a session log.

When implementation state changes, replace stale bullets. Do not append transcripts, chat summaries, command logs, or minor fix notes.
