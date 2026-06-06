"""Dependency-safe video and image-folder primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, cast

import numpy as np
from numpy.typing import NDArray

from atlas3r.contracts.frames import CameraModel, FramePacket

VideoKind = Literal["video_file", "image_directory", "image_file", "missing"]
_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
_IMAGE_SUFFIXES = {".ppm", ".png", ".jpg", ".jpeg"}


class VideoDependencyError(RuntimeError):
    """Raised when frame decoding needs an optional external dependency."""


@dataclass(frozen=True)
class DecodedFrame:
    rgb_u8: NDArray[np.uint8]
    source_uri: str
    original_frame_index: int | None
    timestamp_ns: int
    decoder_name: str


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
        ppm_only = bool(image_files) and all(item.suffix.lower() == ".ppm" for item in image_files)
        return VideoInspection(
            input_path=str(input_path),
            kind="image_directory",
            suffix="",
            byte_size=0,
            frame_count=len(image_files),
            decoder="stdlib_ppm" if ppm_only else "optional_image" if image_files else "none",
            message=(
                "image sequence can be decoded through dependency-free PPM or optional decoders"
                if image_files
                else "directory contains no supported image files"
            ),
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
            decoder="stdlib_ppm" if suffix == ".ppm" else "optional_image",
            message=(
                "single dependency-free PPM image"
                if suffix == ".ppm"
                else "single image requires Pillow, imageio, or OpenCV"
            ),
        )
    if suffix in _VIDEO_SUFFIXES:
        return VideoInspection(
            input_path=str(input_path),
            kind="video_file",
            suffix=suffix,
            byte_size=byte_size,
            frame_count=None,
            decoder="optional_video",
            message="video decoding requires imageio or OpenCV",
        )
    return VideoInspection(
        input_path=str(input_path),
        kind="image_file",
        suffix=suffix,
        byte_size=byte_size,
        frame_count=None,
        decoder="unavailable",
        message="unsupported extension for dependency-safe decoding",
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


def write_ppm_image(path: str | Path, rgb: NDArray[np.uint8]) -> None:
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must be uint8 H,W,3")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(image.shape[0]), int(image.shape[1])
    target.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + image.tobytes())


def decode_input_frames(input_path: str | Path, *, max_frames: int) -> tuple[DecodedFrame, ...]:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    path = Path(input_path)
    inspection = inspect_video_input(path)
    if inspection.kind == "missing":
        raise FileNotFoundError(f"input path does not exist: {path}")
    if inspection.kind == "image_directory":
        image_paths = sorted(
            item for item in path.iterdir() if item.suffix.lower() in _IMAGE_SUFFIXES
        )
        return tuple(
            _decode_image_frame(image_path, frame_id)
            for frame_id, image_path in enumerate(image_paths[:max_frames])
        )
    if inspection.kind == "image_file":
        if path.suffix.lower() in _IMAGE_SUFFIXES:
            return (_decode_image_frame(path, 0),)
        raise VideoDependencyError(f"unsupported image extension for decoding: {path.suffix}")
    if inspection.kind == "video_file":
        return _decode_video_frames(path, max_frames=max_frames)
    raise VideoDependencyError(f"unsupported input kind: {inspection.kind}")


def require_video_decoder() -> None:
    try:
        import imageio.v3  # noqa: F401

        return
    except Exception:
        pass
    try:
        import cv2  # noqa: F401

        return
    except Exception as exc:
        raise VideoDependencyError("video decoding requires imageio or opencv") from exc


def _decode_image_frame(path: Path, frame_id: int) -> DecodedFrame:
    if path.suffix.lower() == ".ppm":
        rgb = _read_ppm(path)
        decoder_name = "stdlib_ppm"
    else:
        rgb, decoder_name = _read_optional_image(path)
    return DecodedFrame(
        rgb_u8=rgb,
        source_uri=str(path),
        original_frame_index=None,
        timestamp_ns=frame_id * 33_333_333,
        decoder_name=decoder_name,
    )


def _decode_video_frames(path: Path, *, max_frames: int) -> tuple[DecodedFrame, ...]:
    imageio_error: Exception | None = None
    try:
        import imageio.v3 as iio

        frames: list[DecodedFrame] = []
        for frame_id, frame in enumerate(iio.imiter(path)):
            if frame_id >= max_frames:
                break
            frames.append(
                DecodedFrame(
                    rgb_u8=_as_rgb_u8(frame),
                    source_uri=str(path),
                    original_frame_index=frame_id,
                    timestamp_ns=frame_id * 33_333_333,
                    decoder_name="imageio.v3",
                )
            )
        if frames:
            return tuple(frames)
    except Exception as exc:
        imageio_error = exc
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise VideoDependencyError(f"OpenCV could not open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        timestamp_step_ns = int(1_000_000_000 / fps) if fps > 0 else 33_333_333
        frames = []
        frame_id = 0
        while frame_id < max_frames:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(
                DecodedFrame(
                    rgb_u8=_as_rgb_u8(rgb),
                    source_uri=str(path),
                    original_frame_index=frame_id,
                    timestamp_ns=frame_id * timestamp_step_ns,
                    decoder_name="opencv",
                )
            )
            frame_id += 1
        capture.release()
        if frames:
            return tuple(frames)
    except Exception as exc:
        if imageio_error is not None:
            raise VideoDependencyError(
                f"video decoder failed; imageio={imageio_error}; opencv={exc}"
            ) from exc
        raise VideoDependencyError(f"video decoder failed: {exc}") from exc
    raise VideoDependencyError(f"video decoder produced no frames: {path}")


def _read_optional_image(path: Path) -> tuple[NDArray[np.uint8], str]:
    pillow_error: Exception | None = None
    try:
        from PIL import Image

        with Image.open(path) as image:
            return _as_rgb_u8(np.asarray(image.convert("RGB"))), "pillow"
    except Exception as exc:
        pillow_error = exc
    try:
        import imageio.v3 as iio

        return _as_rgb_u8(iio.imread(path)), "imageio.v3"
    except Exception as imageio_error:
        try:
            import cv2

            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise VideoDependencyError(f"OpenCV could not read image: {path}")
            return _as_rgb_u8(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), "opencv"
        except Exception as cv2_error:
            raise VideoDependencyError(
                "image decoding requires Pillow, imageio, or OpenCV; "
                f"pillow={pillow_error}; imageio={imageio_error}; opencv={cv2_error}"
            ) from cv2_error


def _as_rgb_u8(value: object) -> NDArray[np.uint8]:
    array = np.asarray(value)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError(f"decoded frame must be H,W,3 or H,W,4, got {array.shape}")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return cast(NDArray[np.uint8], np.ascontiguousarray(array))


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
