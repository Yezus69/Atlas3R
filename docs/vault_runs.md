# Vault runs — the sealed anti-overfitting record

Protocol: `config/vault_assets.json`. The vault scene (TUM freiburg3
long_office_household, a DIFFERENT camera than every fr1 gate scene) is run
ONLY at declared milestones; every run is recorded here with its commit
hash; **nothing is ever tuned in response to a vault result** — a vault
regression is an overfitting ALARM about the development process, never a
bug to fix on the vault scene.

## Run 1 — declared 2026-06-11, milestone: production recipe v3 complete

**Milestone condition:** the v3 production recipe (selector keyframes →
COLMAP pose backend → 2-phase MVS verified depth → perturbation-stability
tier τ=0.005 → 2.5 cm envelope) plus the scale-honest distance instrument
are complete and measured on all fr1 gate scenes (commits `91a9745` …
`99a15d8`). Every threshold, parameter, and policy in this run is FROZEN at
its committed value: selector (0.25 widths, cap 48), k=2, τ=0.005, gate
thresholds (0.30/0.70/0.30 / 0.50/0.25/0.10), envelope files as committed.

**What runs (both recipes, mirroring the fr1 comparison):**
1. M1/M2 on the vault manifest; selector keyframes anchored to the M2
   measured packet ids.
2. Canonical-style recipe: plain MapAnything artifacts at the committed
   envelope (5 cm).
3. v3 recipe: COLMAP sparse (measured fr3 intrinsics pinned, the oracle
   recipe) → pose backend → photometric substrate + keyframe geometric pass
   → disjoint-half A/B passes → stability composite (τ=0.005) at the v2
   envelope (2.5 cm).
4. Teacher + gate on both; scale-honest and Sim(3) distance instruments
   read from the reports. The vault NEVER enters `runs/eval/`.

**Pre-registered expectations (committed before any vault number is
observed):**
- E1. v3 ≥ canonical-style on scale-honest solid_f1_at_5cm and at_10cm
  (the Phase 13 dominance generalizes across cameras).
- E2. v3 scale-honest median solid distance ≤ ~0.10 m (fr1 read
  0.056–0.060 m; cross-camera degradation expected but bounded).
- E3. Gate verdicts coherent with the scene's measured quality (no
  false-accept of a config whose measured placement is far worse than
  fr1-rejected configs).

**ALARM CONDITIONS (defined in advance):**
- A1. v3 WORSE than canonical-style on the vault's scale-honest metrics →
  the recipe campaign overfit fr1; triggers process diagnosis.
- A2. Gate accepts the vault candidate while its measured camera RMSE or
  placement is worse than fr1-rejected configs → gate thresholds overfit
  fr1; triggers gate-process diagnosis.
- A3. Any tool/code change required to make the vault RUN at all is
  recorded as a generalization defect (allowed to fix mechanically, never
  tuned to improve vault METRICS).

**A3 generalization defects found (mechanical fixes only, recorded):**
1. The documented vault invocation had NEVER been runnable: M1's manifest
   validation hard-required the canonical asset ids for ANY manifest. Fixed
   mechanically (the canonical-completeness check now guards only the
   default manifest). The vault protocol was committed without exercising
   its own entry point.
2. The vault's metadata sidecar was staged with a non-M2 schema
   (`camera.*` / `gt_match_tolerance_s` instead of
   `m2_reference.intrinsics.*` / `timestamp_tolerance_s` /
   `pose_translation_units`) — M2 honestly produced zero measured packets.
   Sidecar rewritten to the proven fr1 schema with the published fr3
   calibration values (data staging correction; gitignored data, recorded
   here).
3. Keyframe-only COLMAP staging registered 26/48 keyframes with only 1 of 6
   measured anchors — the SAME production reality phone_room taught
   (Phase 10). The documented production staging (every-8th ∪ keyframes ∪
   anchors, sequential video matching) was applied as the frozen recipe.

**RESULT (run executed 2026-06-11 at commit `debd2b4`; runs/teacher_vault_canonical + runs/teacher_vault_v3; chain: M2 6 measured packets → selector 48 kf (anchored) → MapAnything → COLMAP 366/366 registered, all 6 anchors, sequential matching → 2-phase MVS + A/B → stability composite 18.2% verified):**

| | canonical-style | v3 |
|---|---|---|
| camera Sim(3) RMSE | 1.413 m | 0.825 m |
| estimated scale | 1.152 | **1.064** |
| HONEST solid F1@5cm | 0.171 | **0.231** |
| HONEST solid F1@10cm | 0.367 | **0.564** |
| HONEST precision@10cm | 0.232 | **0.513** |
| HONEST median solid distance | 0.037 m | 0.078 m |
| per_class | 0.791 | 0.736 |
| gate verdict | REJECTED (floor 0.149, fsc 0.622) | REJECTED (floor 0.156, fsc 0.607) |

**NO ALARMS.**
- E1 CONFIRMED: v3 dominance generalizes cross-camera (+35% F1@5cm, +54%
  F1@10cm, 2.2× precision@10cm) — the placement campaign is not fr1
  overfitting.
- E2 CONFIRMED: v3 honest median 0.078 m ≤ the pre-registered 0.10 m bound.
- E3 CONFIRMED — the gate WORKED on an unseen scene: the 2585-frame long
  sweep drifts under sequential matching without loop closure (RMSE
  0.83–1.41 m) and the gate rejected BOTH recipes for the right reasons
  (fsc 0.61–0.62 ≫ 0.25; floor < 0.30). No false accept.
- Honest dips recorded, not explained away: v3 per_class 0.736 < canonical
  0.791 on this scene; canonical's median recall-distance is lower while
  its F1 is far worse (its precision collapses — fewer correct solids).
- The vault names the next yield wall for LONG videos: pose drift without
  loop closure. The gate catches it (the moat holds); turning those
  rejections into accepts needs a loop-closing pose backend (GLOMAP — the
  standing roadmap item), never a weaker gate.
Nothing was tuned in response to any number above. The vault scene remains
sealed and unburned.
