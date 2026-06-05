"""Dependency-safe input and recording primitives."""

from atlas3r.input.metadata import camera_from_focal, scale_camera_model
from atlas3r.input.recording import (
    RecordingFrame,
    RecordingManifest,
    load_recording,
    write_recording,
)
from atlas3r.input.video import (
    VideoDependencyError,
    VideoInspection,
    inspect_video_input,
    load_ppm_sequence_frames,
)

__all__ = [
    "RecordingFrame",
    "RecordingManifest",
    "VideoDependencyError",
    "VideoInspection",
    "camera_from_focal",
    "inspect_video_input",
    "load_ppm_sequence_frames",
    "load_recording",
    "scale_camera_model",
    "write_recording",
]
