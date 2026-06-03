Phase 5D - Train temporal model with teacher confidence weighting.

Start only after Phase 5C external teacher runner bootstrap has passed tests and
been committed. Use the stable Phase 5B/5C teacher-signal cache format as the
training boundary.

Goal: add a small temporal training path that mixes measured TUM RGB-D teacher
signals with validated external pseudo-label teacher signals, weighting losses
by teacher confidence and depth uncertainty. Do not claim benchmark accuracy or
mapping readiness.

Required scope:

- Load measured and external teacher-signal caches through existing validators.
- Build a temporal dataset that aligns clip-cache RGB, intrinsics, poses,
  teacher depth, confidence, valid masks, and `depth_sigma_m`.
- Train the existing tiny temporal model or a minimal successor on center-frame
  depth with confidence/uncertainty weighting.
- Keep truth flags false for accuracy, performance, realtime, and mapping
  readiness unless a named evaluation report proves otherwise.
- Add fixture tests for measured-only, external-only, and mixed-cache batches.
- Run format, lint, typecheck, unit tests, and synthetic fixture smoke relevant
  to teacher-signal training.
