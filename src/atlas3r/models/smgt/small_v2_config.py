"""Configuration for the SMGT-small-v2 diagnostic student."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast


@dataclass(frozen=True)
class SMGTSmallV2Config:
    """Validated shape and architecture settings for SMGT-small-v2."""

    image_height: int = 160
    image_width: int = 224
    clip_length: int = 8
    stem_dim: int = 48
    hidden_dim: int = 96
    feature_dim: int = 128
    memory_dim: int = 160
    pose_hidden_dim: int = 128
    min_depth_m: float = 0.05
    max_depth_m: float = 20.0
    phase_pose_scale: float = 0.75
    phase_pose_max_shift_px: float = 12.0
    learned_pose_residual_scale: float = 0.05

    def __post_init__(self) -> None:
        for field_name in (
            "image_height",
            "image_width",
            "clip_length",
            "stem_dim",
            "hidden_dim",
            "feature_dim",
            "memory_dim",
            "pose_hidden_dim",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name}: must be a positive integer")
        if self.feature_dim < 128:
            raise ValueError("feature_dim: SMGT-small-v2 requires feature_dim >= 128")
        if self.memory_dim < 128:
            raise ValueError("memory_dim: SMGT-small-v2 requires memory_dim >= 128")
        if self.min_depth_m <= 0.0:
            raise ValueError("min_depth_m: must be positive")
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("max_depth_m: must be greater than min_depth_m")
        if self.phase_pose_scale < 0.0:
            raise ValueError("phase_pose_scale: must be non-negative")
        if self.phase_pose_max_shift_px <= 0.0:
            raise ValueError("phase_pose_max_shift_px: must be positive")
        if self.learned_pose_residual_scale < 0.0:
            raise ValueError("learned_pose_residual_scale: must be non-negative")

    def to_json(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> SMGTSmallV2Config:
        if not isinstance(value, dict):
            raise ValueError("SMGTSmallV2Config: config must be a mapping")
        return cls(
            image_height=int(value.get("image_height", 160)),
            image_width=int(value.get("image_width", 224)),
            clip_length=int(value.get("clip_length", 8)),
            stem_dim=int(value.get("stem_dim", 48)),
            hidden_dim=int(value.get("hidden_dim", 96)),
            feature_dim=int(value.get("feature_dim", 128)),
            memory_dim=int(value.get("memory_dim", 160)),
            pose_hidden_dim=int(value.get("pose_hidden_dim", 128)),
            min_depth_m=float(value.get("min_depth_m", 0.05)),
            max_depth_m=float(value.get("max_depth_m", 20.0)),
            phase_pose_scale=float(value.get("phase_pose_scale", 0.75)),
            phase_pose_max_shift_px=float(value.get("phase_pose_max_shift_px", 12.0)),
            learned_pose_residual_scale=float(value.get("learned_pose_residual_scale", 0.05)),
        )


__all__ = ["SMGTSmallV2Config"]
