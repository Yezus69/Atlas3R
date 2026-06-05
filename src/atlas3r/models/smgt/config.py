"""Configuration for the first diagnostic SMGT-tiny student."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast


@dataclass(frozen=True)
class SMGTTinyConfig:
    """Validated SMGT-tiny shape and architecture settings."""

    image_height: int
    image_width: int
    clip_length: int = 8
    hidden_dim: int = 64
    feature_dim: int = 96
    memory_dim: int = 128
    use_temporal_memory: bool = True
    predict_relative_pose: bool = True
    predict_world_pose: bool = True
    min_depth_m: float = 0.05
    max_depth_m: float = 20.0

    def __post_init__(self) -> None:
        for field_name in (
            "image_height",
            "image_width",
            "clip_length",
            "hidden_dim",
            "feature_dim",
            "memory_dim",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name}: must be a positive integer")
        if self.min_depth_m <= 0.0:
            raise ValueError("min_depth_m: must be positive")
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("max_depth_m: must be greater than min_depth_m")
        if not isinstance(self.use_temporal_memory, bool):
            raise ValueError("use_temporal_memory: must be a bool")
        if not isinstance(self.predict_relative_pose, bool):
            raise ValueError("predict_relative_pose: must be a bool")
        if not isinstance(self.predict_world_pose, bool):
            raise ValueError("predict_world_pose: must be a bool")

    def to_json(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> SMGTTinyConfig:
        if not isinstance(value, dict):
            raise ValueError("SMGTTinyConfig: config must be a mapping")
        return cls(
            image_height=int(value["image_height"]),
            image_width=int(value["image_width"]),
            clip_length=int(value.get("clip_length", 8)),
            hidden_dim=int(value.get("hidden_dim", 64)),
            feature_dim=int(value.get("feature_dim", 96)),
            memory_dim=int(value.get("memory_dim", 128)),
            use_temporal_memory=bool(value.get("use_temporal_memory", True)),
            predict_relative_pose=bool(value.get("predict_relative_pose", True)),
            predict_world_pose=bool(value.get("predict_world_pose", True)),
            min_depth_m=float(value.get("min_depth_m", 0.05)),
            max_depth_m=float(value.get("max_depth_m", 20.0)),
        )


__all__ = ["SMGTTinyConfig"]
