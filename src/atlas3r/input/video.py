"""Dependency-safe video and image-folder primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
from numpy.typing import NDArray

from atlas3r.contracts.frames import CameraModel, FramePacket

VideoKind = Literal["video_file", "image_directory", "image_file", "missing"]
_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
_IMAGE_SUFFIXES = {".ppm"}


class VideoDependencyError(RuntimeError):
    """Raised when frame decoding needs an optional external dependency."""


@dataclass(frozen=True)
class VideoInspection:
    input_path: str
    kind: VideoKind
    suffix: str
    byte_size: int
    frame_count: int | None
    decoder: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "input_path": self.input_path,
            "kind": self.kind,
            "suffix": self.suffix,
            "byte_size": self.byte_size,
            "frame_count": self.frame_count,
            "decoder": self.decoder,
            "message": self.message,
        }


def inspect_video_input(path: str | Path) -> VideoInspection:
    input_path = Path(path)
    if not input_path.exists():
        return VideoInspection(
            input_path=str(input_path),
            kind="missing",
            suffix=input_path.suffix.lower(),
            byte_size=0,
            frame_count=None,
            decoder="none",
            message="input path does not exist",
        )
    if input_path.is_dir():
        image_files = sorted(p for p in input_path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
        return VideoInspection(
            input_path=str(input_path),
            kind="image_directory",
            suffix="",
            byte_size=0,
            frame_count=len(image_files),
            decoder="stdlib_ppm" if image_files else "none",
            message="PPM sequence is dependency-free; video decoding is not attempted",
        )
    suffix = input_path.suffix.lower()
    byte_size = input_path.stat().st_size
    if suffix in _IMAGE_SUFFIXES:
        return VideoInspection(
            input_path=str(input_path),
            kind="image_file",
            suffix=suffix,
            byte_size=byte_size,
            frame_count=1,
            decoder="stdlib_ppm",
            message="single dependency-free PPM image",
        )
    if suffix in _VIDEO_SUFFIXES:
        return VideoInspection(
            input_path=str(input_path),
            kind="video_file",
            suffix=suffix,
            byte_size=byte_size,
            frame_count=None,
            decoder="unavailable",
            message="install imageio or opencv in a future adapter to decode frames",
        )
    return VideoInspection(
        input_path=str(input_path),
        kind="image_file",
        suffix=suffix,
        byte_size=byte_size,
        frame_count=None,
        decoder="unavailable",
        message="unsupported extension for dependency-free decoding",
    )


def load_ppm_sequence_frames(
    directory: str | Path, camera: CameraModel, *, timestamp_step_ns: int = 33_333_333
) -> tuple[FramePacket, ...]:
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"PPM sequence path is not a directory: {root}")
    paths = sorted(p for p in root.iterdir() if p.suffix.lower() == ".ppm")
    frames: list[FramePacket] = []
    for frame_id, ppm_path in enumerate(paths):
        rgb = _read_ppm(ppm_path)
        if rgb.shape[0] != camera.height or rgb.shape[1] != camera.width:
            raise ValueError(f"{ppm_path} dimensions do not match camera")
        frames.append(
            FramePacket(
                frame_id=frame_id,
                timestamp_ns=frame_id * timestamp_step_ns,
                rgb_u8=rgb,
                K_original=camera.K,
                K_model=camera.K,
                resize_transform=np.eye(3, dtype=np.float32),
                camera_metadata={"source_format": "ppm_sequence"},
                source_uri=str(ppm_path),
            )
        )
    if not frames:
        raise ValueError(f"no .ppm frames found in {root}")
    return tuple(frames)


def read_ppm_image(path: str | Path) -> NDArray[np.uint8]:
    return _read_ppm(Path(path))


def require_video_decoder() -> None:
    raise VideoDependencyError("video decoding is not bundled; install imageio or opencv later")


def _read_ppm(path: Path) -> NDArray[np.uint8]:
    with path.open("rb") as handle:
        magic = _next_token(handle)
        if magic != b"P6":
            raise ValueError(f"{path} is not a binary P6 PPM file")
        width = int(_next_token(handle))
        height = int(_next_token(handle))
        max_value = int(_next_token(handle))
        if width <= 0 or height <= 0 or max_value != 255:
            raise ValueError(f"{path} has unsupported PPM dimensions or max value")
        payload = handle.read()
    expected = width * height * 3
    if len(payload) != expected:
        raise ValueError(f"{path} payload length mismatch: expected {expected}, got {len(payload)}")
    return np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 3)).copy()


def _next_token(handle: BinaryIO) -> bytes:
    token = bytearray()
    while True:
        char = handle.read(1)
        if char == b"":
            raise ValueError("unexpected end of PPM header")
        if char == b"#":
            handle.readline()
            continue
        if char.isspace():
            if token:
                return bytes(token)
            continue
        token.extend(char)
