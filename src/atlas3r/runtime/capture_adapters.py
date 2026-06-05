"""Dependency-safe live capture adapter boundary."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from atlas3r.api import FramePacket
from atlas3r.recording.schema import Atlas3RRecording, load_recording


class CaptureAdapterError(RuntimeError):
    """Base error for runtime capture adapter failures."""


class CaptureAdapterDependencyError(CaptureAdapterError):
    """Raised when an optional capture dependency is unavailable."""

    def __init__(self, status: CaptureAdapterStatus) -> None:
        detail = status.reason or "optional dependency unavailable"
        hint = "" if status.install_hint is None else f" {status.install_hint}"
        super().__init__(f"{status.display_name} capture adapter is unavailable: {detail}.{hint}")
        self.status = status


class CaptureAdapterConfigError(CaptureAdapterError):
    """Raised when capture adapter configuration or hardware is unavailable."""


@dataclass(frozen=True)
class CaptureAdapterStatus:
    name: str
    display_name: str
    available: bool
    reason: str | None = None
    install_hint: str | None = None
    capabilities: Mapping[str, object] = field(default_factory=dict)
    dependency_versions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name: must be non-empty")
        if not self.display_name:
            raise ValueError("display_name: must be non-empty")
        if not isinstance(self.available, bool):
            raise ValueError("available: must be a bool")
        if self.reason is not None and not self.reason:
            raise ValueError("reason: must be non-empty when present")
        if self.install_hint is not None and not self.install_hint:
            raise ValueError("install_hint: must be non-empty when present")
        object.__setattr__(self, "capabilities", dict(self.capabilities))
        object.__setattr__(self, "dependency_versions", dict(self.dependency_versions))

    def to_json_record(self) -> dict[str, object]:
        return {
            "available": self.available,
            "capabilities": dict(self.capabilities),
            "dependency_versions": dict(self.dependency_versions),
            "display_name": self.display_name,
            "install_hint": self.install_hint,
            "name": self.name,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CaptureFrame:
    frame_id: int
    timestamp_ns: int
    frame_packet: FramePacket | None
    source_metadata: Mapping[str, object] = field(default_factory=dict)
    has_measured_depth: bool = False
    has_measured_pose: bool = False
    measured_depth_path: str | None = None

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id: must be non-negative")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns: must be non-negative")
        if self.frame_packet is not None and self.frame_packet.frame_id != self.frame_id:
            raise ValueError("frame_packet.frame_id: must match CaptureFrame frame_id")
        if self.measured_depth_path is not None and not self.measured_depth_path:
            raise ValueError("measured_depth_path: must be non-empty when present")
        object.__setattr__(self, "source_metadata", dict(self.source_metadata))


class CaptureFrameStream(Protocol):
    def frames(self) -> Iterator[CaptureFrame]: ...


class LiveCameraAdapter(Protocol):
    def status(self) -> CaptureAdapterStatus: ...
    def open(self, config: Mapping[str, object]) -> AbstractContextManager[CaptureFrameStream]: ...


class OpenCVCameraAdapter:
    """OpenCV capture adapter with lazy optional dependency loading."""

    name = "opencv-camera"
    display_name = "OpenCV Camera"
    install_hint = "Install OpenCV with `python -m pip install opencv-python`."

    def status(self) -> CaptureAdapterStatus:
        if find_spec("cv2") is None:
            return CaptureAdapterStatus(
                name=self.name,
                display_name=self.display_name,
                available=False,
                reason="missing optional dependency: cv2",
                install_hint=self.install_hint,
                capabilities=_opencv_capabilities(),
            )
        return CaptureAdapterStatus(
            name=self.name,
            display_name=self.display_name,
            available=True,
            capabilities=_opencv_capabilities(),
        )

    def open(self, config: Mapping[str, object]) -> AbstractContextManager[CaptureFrameStream]:
        camera_index = _int_config(config, "camera_index", default=0)
        if camera_index < 0:
            raise CaptureAdapterConfigError("camera_index: must be non-negative")
        status = self.status()
        if not status.available:
            raise CaptureAdapterDependencyError(status)
        width = _optional_positive_int_config(config, "width")
        height = _optional_positive_int_config(config, "height")
        target_fps = _optional_positive_float_config(config, "target_fps")
        return _OpenCVCameraStream(
            camera_index=camera_index,
            width=width,
            height=height,
            target_fps=target_fps,
            K=_optional_intrinsics(config),
        )


class ReplayRecordingAdapter:
    """Expose a validated Atlas3R recording as a live-like capture stream."""

    name = "replay-recording"
    display_name = "Atlas3R Recording Replay"

    def status(self) -> CaptureAdapterStatus:
        return CaptureAdapterStatus(
            name=self.name,
            display_name=self.display_name,
            available=True,
            capabilities={
                "source": "atlas3r_recording",
                "supports_measured_depth_metadata": True,
                "supports_measured_pose_metadata": True,
                "supports_rgb": True,
                "supports_streaming": True,
            },
        )

    def open(self, config: Mapping[str, object]) -> AbstractContextManager[CaptureFrameStream]:
        value = config.get("recording")
        if not isinstance(value, str | Path):
            raise CaptureAdapterConfigError("recording: must be a path")
        return _ReplayRecordingStream(Path(value))


class _ReplayRecordingStream:
    def __init__(self, recording_path: Path) -> None:
        self._recording_path = recording_path
        self._recording: Atlas3RRecording | None = None

    def __enter__(self) -> _ReplayRecordingStream:
        self._recording = load_recording(self._recording_path)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._recording = None

    def frames(self) -> Iterator[CaptureFrame]:
        if self._recording is None:
            raise CaptureAdapterConfigError("ReplayRecordingAdapter must be opened before frames()")
        for frame in self._recording.frames:
            yield CaptureFrame(
                frame_id=frame.frame_id,
                timestamp_ns=int(round(frame.timestamp_s * 1_000_000_000.0)),
                frame_packet=None,
                source_metadata={
                    "adapter": ReplayRecordingAdapter.name,
                    "recording": str(self._recording.root),
                    "source_metadata": dict(frame.source_metadata),
                },
                has_measured_depth=frame.depth_path is not None,
                has_measured_pose=frame.T_world_camera is not None,
                measured_depth_path=frame.depth_path,
            )


class _OpenCVCameraStream:
    def __init__(
        self,
        *,
        camera_index: int,
        width: int | None,
        height: int | None,
        target_fps: float | None,
        K: npt.NDArray[np.float32] | None,
    ) -> None:
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._target_fps = target_fps
        self._K = K
        self._capture: Any | None = None
        self._cv2: Any | None = None

    def __enter__(self) -> _OpenCVCameraStream:
        import cv2

        self._cv2 = cv2
        capture = cv2.VideoCapture(self._camera_index)
        if self._width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        if self._height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
        if self._target_fps is not None:
            capture.set(cv2.CAP_PROP_FPS, float(self._target_fps))
        if not bool(capture.isOpened()):
            capture.release()
            raise CaptureAdapterConfigError(
                f"OpenCV camera {self._camera_index} could not be opened"
            )
        self._capture = capture
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._cv2 = None

    def frames(self) -> Iterator[CaptureFrame]:
        if self._capture is None:
            raise CaptureAdapterConfigError("OpenCVCameraAdapter must be opened before frames()")
        frame_id = 0
        while True:
            ok, bgr = self._capture.read()
            if not ok:
                break
            rgb = np.asarray(bgr[:, :, ::-1], dtype=np.uint8)
            height, width = int(rgb.shape[0]), int(rgb.shape[1])
            K = self._K if self._K is not None else _fallback_intrinsics(width, height)
            packet = FramePacket(
                frame_id=frame_id,
                timestamp_ns=time.time_ns(),
                rgb_u8=rgb,
                rgb_model=rgb.transpose(2, 0, 1).astype(np.float32) / np.float32(255.0),
                K_original=K.astype(np.float32, copy=True),
                K_model=K.astype(np.float32, copy=True),
                distortion=None,
                resize_transform=np.eye(3, dtype=np.float32),
                camera_metadata={
                    "adapter": OpenCVCameraAdapter.name,
                    "intrinsics_source": "config_or_fallback",
                },
            )
            yield CaptureFrame(
                frame_id=frame_id,
                timestamp_ns=packet.timestamp_ns,
                frame_packet=packet,
                source_metadata={"adapter": OpenCVCameraAdapter.name},
            )
            frame_id += 1


def list_capture_adapters() -> tuple[CaptureAdapterStatus, ...]:
    adapters: tuple[LiveCameraAdapter, ...] = (OpenCVCameraAdapter(), ReplayRecordingAdapter())
    return tuple(adapter.status() for adapter in adapters)


def _opencv_capabilities() -> dict[str, object]:
    return {
        "supports_rgb": True,
        "supports_streaming": True,
        "supports_measured_depth_metadata": False,
        "supports_measured_pose_metadata": False,
    }


def _fallback_intrinsics(width: int, height: int) -> npt.NDArray[np.float32]:
    focal = np.float32(max(width, height))
    return np.asarray(
        [[focal, 0.0, (width - 1) * 0.5], [0.0, focal, (height - 1) * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _optional_intrinsics(config: Mapping[str, object]) -> npt.NDArray[np.float32] | None:
    value = config.get("K")
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise CaptureAdapterConfigError("K: must be a finite 3x3 intrinsics matrix")
    return array


def _int_config(config: Mapping[str, object], key: str, *, default: int) -> int:
    value = config.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise CaptureAdapterConfigError(f"{key}: must be an integer")
    return value


def _optional_positive_int_config(config: Mapping[str, object], key: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CaptureAdapterConfigError(f"{key}: must be a positive integer when present")
    return value


def _optional_positive_float_config(config: Mapping[str, object], key: str) -> float | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool) or float(value) <= 0.0:
        raise CaptureAdapterConfigError(f"{key}: must be a positive number when present")
    return float(value)


__all__ = [
    "CaptureAdapterConfigError",
    "CaptureAdapterDependencyError",
    "CaptureAdapterError",
    "CaptureAdapterStatus",
    "CaptureFrame",
    "CaptureFrameStream",
    "LiveCameraAdapter",
    "OpenCVCameraAdapter",
    "ReplayRecordingAdapter",
    "list_capture_adapters",
]
