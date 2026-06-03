Phase 5C - External Teacher Runner Bootstrap. Add dependency-isolated local
external runners for one depth teacher and one multi-view geometry teacher,
starting with Depth Pro and VGGT if install paths are available. Runners must
write Atlas3R teacher-signal caches only. Do not vendor repos/weights; do not
change base dependencies; do not train a new model until teacher-signal
validation passes.
