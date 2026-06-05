"""Truth-boundary labels for measured, anchored, synthetic, and pseudo data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

TruthLabelType = Literal[
    "measured_gt",
    "anchored_capture",
    "synthetic_gt",
    "cad_aligned_approx",
    "teacher_pseudo",
    "unanchored_mp4_pseudo",
    "debug_synthetic",
    "unknown",
]

_TRUTH_LABELS: set[str] = {
    "measured_gt",
    "anchored_capture",
    "synthetic_gt",
    "cad_aligned_approx",
    "teacher_pseudo",
    "unanchored_mp4_pseudo",
    "debug_synthetic",
    "unknown",
}


@dataclass(frozen=True)
class TruthBoundary:
    """Explicitly states what kind of geometry or labels a record may claim."""

    label_type: TruthLabelType
    metric_scale_source: str
    measured_geometry: bool = False
    observed_only: bool = True
    predicted_completion: bool = False
    hidden_geometry_measured: bool = False
    accuracy_report: bool = False
    realtime_claim: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.label_type not in _TRUTH_LABELS:
            raise ValueError(f"unknown truth label_type: {self.label_type}")
        if not self.metric_scale_source:
            raise ValueError("metric_scale_source must be non-empty")
        if self.hidden_geometry_measured and not self.measured_geometry:
            raise ValueError("hidden geometry cannot be measured when measured_geometry=false")
        if (
            self.label_type in {"teacher_pseudo", "unanchored_mp4_pseudo", "debug_synthetic"}
            and self.measured_geometry
        ):
            raise ValueError("pseudo labels cannot claim measured geometry")
        if self.predicted_completion and self.observed_only:
            raise ValueError("predicted completion cannot also be observed-only")

    @classmethod
    def unanchored_mp4(cls, notes: str = "") -> TruthBoundary:
        return cls(
            label_type="unanchored_mp4_pseudo",
            metric_scale_source="unanchored_rgb_prior",
            measured_geometry=False,
            observed_only=True,
            notes=notes,
        )

    @classmethod
    def teacher_pseudo(cls, scale_source: str, notes: str = "") -> TruthBoundary:
        return cls(
            label_type="teacher_pseudo",
            metric_scale_source=scale_source,
            measured_geometry=False,
            observed_only=True,
            notes=notes,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "label_type": self.label_type,
            "metric_scale_source": self.metric_scale_source,
            "measured_geometry": self.measured_geometry,
            "observed_only": self.observed_only,
            "predicted_completion": self.predicted_completion,
            "hidden_geometry_measured": self.hidden_geometry_measured,
            "accuracy_report": self.accuracy_report,
            "realtime_claim": self.realtime_claim,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TruthBoundary:
        label_type = str(data.get("label_type", "unknown"))
        if label_type not in _TRUTH_LABELS:
            raise ValueError(f"unknown truth label_type: {label_type}")
        return cls(
            label_type=cast(TruthLabelType, label_type),
            metric_scale_source=str(data.get("metric_scale_source", "unknown")),
            measured_geometry=bool(data.get("measured_geometry", False)),
            observed_only=bool(data.get("observed_only", True)),
            predicted_completion=bool(data.get("predicted_completion", False)),
            hidden_geometry_measured=bool(data.get("hidden_geometry_measured", False)),
            accuracy_report=bool(data.get("accuracy_report", False)),
            realtime_claim=bool(data.get("realtime_claim", False)),
            notes=str(data.get("notes", "")),
        )
