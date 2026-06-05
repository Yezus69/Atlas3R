"""Confidence/sigma/dynamic valid-mask policy for RGB student mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

STUDENT_MAP_VALID_POLICIES = ("confidence", "confidence_sigma", "all_positive")


@dataclass(frozen=True)
class StudentMapGateConfig:
    """Resolved per-pixel mapping gate for SMGT-tiny predictions."""

    min_depth_m: float
    max_depth_m: float | None = None
    confidence_threshold: float = 0.30
    max_sigma_m: float | None = None
    dynamic_threshold: float = 0.50
    policy: str = "confidence"

    def __post_init__(self) -> None:
        if self.policy not in STUDENT_MAP_VALID_POLICIES:
            raise ValueError(
                "student_map_valid_policy: must be confidence, confidence_sigma, or all_positive"
            )
        if self.min_depth_m <= 0.0 or not np.isfinite(self.min_depth_m):
            raise ValueError("student_min_depth_m: must be positive and finite")
        if self.max_depth_m is not None and (
            self.max_depth_m <= self.min_depth_m or not np.isfinite(self.max_depth_m)
        ):
            raise ValueError(
                "student_max_depth_m: must be finite and greater than student_min_depth_m"
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("student_confidence_threshold: must be in [0, 1]")
        if self.max_sigma_m is not None and (
            self.max_sigma_m <= 0.0 or not np.isfinite(self.max_sigma_m)
        ):
            raise ValueError("student_max_sigma_m: must be positive and finite")
        if not 0.0 <= self.dynamic_threshold <= 1.0:
            raise ValueError("student_dynamic_threshold: must be in [0, 1]")

    @property
    def all_positive_policy_unsafe(self) -> bool:
        return self.policy == "all_positive"

    def to_json(self) -> dict[str, object]:
        return {
            "student_min_depth_m": self.min_depth_m,
            "student_max_depth_m": self.max_depth_m,
            "student_confidence_threshold": self.confidence_threshold,
            "student_max_sigma_m": self.max_sigma_m,
            "student_dynamic_threshold": self.dynamic_threshold,
            "student_map_valid_policy": self.policy,
            "student_map_valid_policy_unsafe": self.all_positive_policy_unsafe,
        }


@dataclass(frozen=True)
class StudentMapGateResult:
    """Mask and count/rate diagnostics for one predicted frame."""

    valid_mask: npt.NDArray[np.bool_]
    stats: dict[str, object]


def resolve_student_map_valid_policy(requested: str | None, max_sigma_m: float | None) -> str:
    """Resolve the default policy from whether a sigma threshold is active."""

    if requested is not None:
        if requested not in STUDENT_MAP_VALID_POLICIES:
            raise ValueError(
                "student_map_valid_policy: must be confidence, confidence_sigma, or all_positive"
            )
        return requested
    return "confidence_sigma" if max_sigma_m is not None else "confidence"


def student_mapping_valid_mask(
    *,
    depth_m: npt.NDArray[Any],
    depth_sigma_m: npt.NDArray[Any],
    confidence: npt.NDArray[Any],
    dynamic_probability: npt.NDArray[Any],
    config: StudentMapGateConfig,
    existing_static_mask: npt.NDArray[Any] | None = None,
) -> StudentMapGateResult:
    """Return the conservative mapper mask and diagnostics for one frame."""

    depth = np.asarray(depth_m)
    sigma = np.asarray(depth_sigma_m)
    conf = np.asarray(confidence)
    dynamic = np.asarray(dynamic_probability)
    if depth.shape != sigma.shape or depth.shape != conf.shape or depth.shape != dynamic.shape:
        raise ValueError("student gate inputs must share HxW shape")
    total = int(depth.size)
    raw_valid = np.isfinite(depth) & (depth > 0.0)
    depth_range_valid = raw_valid & (depth >= config.min_depth_m)
    if config.max_depth_m is not None:
        depth_range_valid &= depth <= config.max_depth_m
    confidence_valid = depth_range_valid & np.isfinite(conf) & (conf >= config.confidence_threshold)
    sigma_valid = confidence_valid & np.isfinite(sigma)
    if config.max_sigma_m is not None:
        sigma_valid &= sigma <= config.max_sigma_m
    dynamic_ok = np.isfinite(dynamic) & (dynamic < config.dynamic_threshold)
    if config.policy == "all_positive":
        policy_base = raw_valid
        valid = raw_valid.copy()
    elif config.policy == "confidence":
        policy_base = confidence_valid
        valid = confidence_valid & dynamic_ok
    else:
        policy_base = sigma_valid
        valid = sigma_valid & dynamic_ok
    if existing_static_mask is not None:
        static = np.asarray(existing_static_mask)
        if static.shape != depth.shape:
            raise ValueError("existing_static_mask must share HxW shape")
        valid &= static.astype(bool)
    mapped_count = int(np.count_nonzero(valid))
    stats = {
        **config.to_json(),
        "raw_valid_pixel_count": int(np.count_nonzero(raw_valid)),
        "depth_range_valid_pixel_count": int(np.count_nonzero(depth_range_valid)),
        "confidence_gated_valid_pixel_count": int(np.count_nonzero(confidence_valid)),
        "sigma_gated_valid_pixel_count": int(np.count_nonzero(sigma_valid)),
        "dynamic_rejected_pixel_count": int(np.count_nonzero(policy_base & ~dynamic_ok)),
        "mapped_pixel_count": mapped_count,
        "total_pixel_count": total,
        "raw_valid_pixel_ratio": _ratio(raw_valid, total),
        "confidence_gated_valid_pixel_ratio": _ratio(confidence_valid, total),
        "sigma_gated_valid_pixel_ratio": _ratio(sigma_valid, total),
        "dynamic_rejected_pixel_ratio": _count_ratio(
            int(np.count_nonzero(policy_base & ~dynamic_ok)), total
        ),
        "mapped_pixel_ratio": _count_ratio(mapped_count, total),
    }
    if config.all_positive_policy_unsafe:
        stats["student_map_valid_policy_warning"] = (
            "all_positive is diagnostic-only and unsafe for student RGB mapping"
        )
    return StudentMapGateResult(valid_mask=valid.astype(np.bool_, copy=False), stats=stats)


def sanitize_student_depth(depth_m: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """Return finite non-negative depth values for the DepthObservation contract."""

    depth = np.asarray(depth_m, dtype=np.float32)
    return np.where(np.isfinite(depth) & (depth >= 0.0), depth, 0.0).astype(np.float32)


def sanitize_student_sigma(
    depth_sigma_m: npt.NDArray[Any], *, fallback_m: float = 1.0
) -> npt.NDArray[np.float32]:
    """Return finite non-negative sigma values for the DepthObservation contract."""

    sigma = np.asarray(depth_sigma_m, dtype=np.float32)
    fallback = np.float32(max(float(fallback_m), 1e-6))
    return np.where(np.isfinite(sigma) & (sigma >= 0.0), sigma, fallback).astype(np.float32)


def sanitize_student_confidence(confidence: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """Return finite confidence values clipped to [0, 1]."""

    conf = np.asarray(confidence, dtype=np.float32)
    conf = np.where(np.isfinite(conf), conf, 0.0)
    return np.clip(conf, 0.0, 1.0).astype(np.float32)


def aggregate_student_gate_stats(stats: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate per-frame gate stats using pixel-count denominators."""

    if not stats:
        return {
            "raw_valid_pixel_ratio": 0.0,
            "confidence_gated_valid_pixel_ratio": 0.0,
            "sigma_gated_valid_pixel_ratio": 0.0,
            "dynamic_rejected_pixel_ratio": 0.0,
            "mapped_pixel_ratio": 0.0,
        }
    total = sum(_stat_int(item, "total_pixel_count") for item in stats)
    count_keys = (
        "raw_valid_pixel_count",
        "depth_range_valid_pixel_count",
        "confidence_gated_valid_pixel_count",
        "sigma_gated_valid_pixel_count",
        "dynamic_rejected_pixel_count",
        "mapped_pixel_count",
    )
    totals = {key: sum(_stat_int(item, key) for item in stats) for key in count_keys}
    first = stats[0]
    aggregate = {
        key: first[key]
        for key in (
            "student_min_depth_m",
            "student_max_depth_m",
            "student_confidence_threshold",
            "student_max_sigma_m",
            "student_dynamic_threshold",
            "student_map_valid_policy",
            "student_map_valid_policy_unsafe",
        )
    }
    aggregate.update(totals)
    aggregate["total_pixel_count"] = total
    aggregate["raw_valid_pixel_ratio"] = _count_ratio(totals["raw_valid_pixel_count"], total)
    aggregate["confidence_gated_valid_pixel_ratio"] = _count_ratio(
        totals["confidence_gated_valid_pixel_count"], total
    )
    aggregate["sigma_gated_valid_pixel_ratio"] = _count_ratio(
        totals["sigma_gated_valid_pixel_count"], total
    )
    aggregate["dynamic_rejected_pixel_ratio"] = _count_ratio(
        totals["dynamic_rejected_pixel_count"], total
    )
    aggregate["mapped_pixel_ratio"] = _count_ratio(totals["mapped_pixel_count"], total)
    if bool(first.get("student_map_valid_policy_unsafe", False)):
        aggregate["student_map_valid_policy_warning"] = first["student_map_valid_policy_warning"]
    return aggregate


def _stat_int(item: dict[str, object], key: str) -> int:
    value = item.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key}: gate stat must be an integer")
    return value


def _ratio(mask: npt.NDArray[np.bool_], total: int) -> float:
    return _count_ratio(int(np.count_nonzero(mask)), total)


def _count_ratio(count: int, total: int) -> float:
    return float(count / max(total, 1))


__all__ = [
    "STUDENT_MAP_VALID_POLICIES",
    "StudentMapGateConfig",
    "StudentMapGateResult",
    "aggregate_student_gate_stats",
    "resolve_student_map_valid_policy",
    "sanitize_student_confidence",
    "sanitize_student_depth",
    "sanitize_student_sigma",
    "student_mapping_valid_mask",
]
