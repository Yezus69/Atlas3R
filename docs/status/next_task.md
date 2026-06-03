Phase 5C - External Teacher Runner Bootstrap. Run Phase 5B.1 cleanup first.
Add dependency-isolated local external runners for one depth teacher and one
multi-view geometry teacher, starting with Depth Pro and VGGT if install paths
are available. Runners must emit Atlas3R teacher-signal caches only. Do not
vendor repos or weights; do not change base dependencies; do not train until
teacher outputs validate through ingest, inspect, and map diagnostics.
