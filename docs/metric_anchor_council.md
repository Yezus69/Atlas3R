# Metric Anchor Council Evidence

Scope: learned metric-depth anchors already staged on disk. The council does
not run external models and does not use measured GT to build candidate output.
Same-artifact primary anchors are self-audits only. Independent learned anchors
are soft `ScaleEvidence`; they never promote `measured_metric`.

Diagnostic command:

```text
python tools/run_metric_anchor_council.py
```

Output:

```text
runs/_diag/metric_anchor_council_report.json
```

Measured result after excluding same-artifact self-anchors from consensus:

- `reference_metric`: raw independent DA3 consensus scale 0.760 vs true
  Sim(3) scale 1.028; predicted relative uncertainty 0.060 did not cover
  log-scale error 0.303.
- `reference_metric_desk`: raw independent DA3 consensus scale 0.713 vs true
  Sim(3) scale 1.083; predicted relative uncertainty 0.085 did not cover
  log-scale error 0.417.
- Because both calibration scenes under-covered, `CALIBRATED_LEARNED_SCALE_ENABLED`
  remains `False`. Raw learned-anchor scales are reported under
  `raw_uncalibrated_consensus`; the teacher receives only scale 1.0 with high
  uncertainty, or no soft scale evidence when all anchors reject.
- `reference_metric_room`: no usable independent anchor; the primary artifact
  is a self-audit and rejected on frame-scale instability 0.308, and DA3 anchors
  were absent.
- `phone_room`: no usable independent anchor; the primary artifact is a
  self-audit and rejected on frame-scale instability 0.446, and DA3 backup
  rejected on instability 0.264, cross-view residual 0.700, and inbounds ratio
  0.015.

Residual-monotone risk certificate:

- Global metric scale is unobservable from monocular RGB alone. A learned
  metric anchor can only be trusted inside a measured residual-support envelope.
- Each independent usable anchor emits a residual risk score from normalized
  frame-scale instability, single-frame outliers, cross-view log residual,
  inbounds deficit, and sample deficit.
- The diagnostic builds a monotone envelope from measured calibration scenes:
  a target can borrow only from calibration rows whose residual risk is at least
  as hard. Outside that support, the result is `no_transfer_authority`.
- Current calibration has only 2 usable measured rows, below the promotion floor
  of 8, so the certificate status is `disabled_calibration_underpowered`.
- Current rows: `reference_metric` risk 0.246, observed raw log-scale error
  0.303; `reference_metric_desk` risk 0.567, observed raw log-scale error
  0.417. `reference_metric_room` and `phone_room` have no usable independent
  anchor, so both receive `no_transfer_authority`.

Trainability audit:

```text
python tools/run_trainability_falsifier.py
```

Current 5 cm dangerous-free rates remain too high for training-grade robot
occupancy: `reference_metric` 0.327 and `reference_metric_desk` 0.567. The
teacher is therefore gaining rejection authority, not trainable phone-room
yield, from this slice.
