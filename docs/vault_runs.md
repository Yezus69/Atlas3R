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

**Result:** (recorded below after the run; the run happens once.)
