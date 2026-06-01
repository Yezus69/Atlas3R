"""Shared constants for the TeacherPrediction cache format."""

CACHE_FORMAT_NAME = "atlas3r_teacher_prediction_cache"
CACHE_FORMAT_VERSION = 1
FRAME_SUMMARIES_PATH = "frame_summaries.jsonl"
OPTIONAL_ARRAY_KEYS = (
    "depth_m",
    "depth_sigma_m",
    "normal_camera",
    "point_world",
    "confidence",
    "static_mask",
    "object_embeddings",
    "object_mask_logits",
)

__all__ = [
    "CACHE_FORMAT_NAME",
    "CACHE_FORMAT_VERSION",
    "FRAME_SUMMARIES_PATH",
    "OPTIONAL_ARRAY_KEYS",
]
