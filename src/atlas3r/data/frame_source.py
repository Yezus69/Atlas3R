"""Dependency-free RGB frame sources that emit Atlas3R FramePacket records."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from atlas3r.api import FramePacket
from atlas3r.api.validation import validate_finite_numeric_array, validate_intrinsics

Array = npt.NDArray[Any]
PathLike = str | Path

FRAME_SOURCE_SMOKE_NPZ = "frame_source_smoke.npz"
PPM_INTRINSICS_SIDECAR = "intrinsics.npz"


class RGBFrameSource(Protocol):
    def frames(self) -> Iterator[FramePacket]:
        """Yield validated RGB frame packets in deterministic order."""


@dataclass(frozen=True)
class NPZFrameSource:
    path: PathLike
    frame_id_start: int = 0

    def frames(self) -> Iterator[FramePacket]:
        yield from load_npz_clip_frames(self.path, frame_id_start=self.frame_id_start)


@dataclass(frozen=True)
class PPMSequenceFrameSource:
    directory: PathLike
    K: Array | None = None
    intrinsics_sidecar: PathLike | None = None
    frame_id_start: int = 0

    def frames(self) -> Iterator[FramePacket]:
        yield from load_ppm_sequence_frames(
            self.directory,
            K=self.K,
            intrinsics_sidecar=self.intrinsics_sidecar,
            frame_id_start=self.frame_id_start,
        )


def load_npz_clip_frames(path: PathLike, *, frame_id_start: int = 0) -> tuple[FramePacket, ...]:
    clip_path = Path(path)
    with np.load(clip_path, allow_pickle=False) as clip:
        if "rgb_u8" not in clip.files:
            raise ValueError(f"{clip_path}: missing required rgb_u8 array")
        if "K" not in clip.files:
            raise ValueError(f"{clip_path}: missing required K array")
        rgb_u8 = _validate_rgb_clip(f"{clip_path}: rgb_u8", clip["rgb_u8"])
        intrinsics = _validate_clip_intrinsics(f"{clip_path}: K", clip["K"], rgb_u8.shape[0])

    return _frames_from_rgb_clip(
        rgb_u8,
        intrinsics,
        frame_id_start=frame_id_start,
        source_format="npz",
        source_path=clip_path,
    )


def load_ppm_sequence_frames(
    directory: PathLike,
    *,
    K: Array | None = None,
    intrinsics_sidecar: PathLike | None = None,
    frame_id_start: int = 0,
) -> tuple[FramePacket, ...]:
    sequence_dir = Path(directory)
    if not sequence_dir.is_dir():
        raise ValueError(f"{sequence_dir}: must be a directory")

    ppm_paths = tuple(sorted(sequence_dir.glob("*.ppm")))
    if not ppm_paths:
        raise ValueError(f"{sequence_dir}: no .ppm files found")

    frames = tuple(_read_ppm_p6(path) for path in ppm_paths)
    first_shape = frames[0].shape
    for path, frame in zip(ppm_paths, frames, strict=True):
        if frame.shape != first_shape:
            raise ValueError(f"{path}: PPM frame shape must match first frame {first_shape}")
    rgb_u8 = _validate_rgb_clip(f"{sequence_dir}: ppm_sequence", np.stack(frames, axis=0))
    intrinsics_value = (
        K if K is not None else _load_intrinsics_sidecar(sequence_dir, intrinsics_sidecar)
    )
    intrinsics = _validate_clip_intrinsics(
        f"{sequence_dir}: K",
        intrinsics_value,
        rgb_u8.shape[0],
    )

    return _frames_from_rgb_clip(
        rgb_u8,
        intrinsics,
        frame_id_start=frame_id_start,
        source_format="ppm_sequence",
        source_path=sequence_dir,
        source_files=ppm_paths,
    )


def write_frame_source_smoke_fixture(output_dir: PathLike) -> tuple[FramePacket, ...]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    clip_path = target / FRAME_SOURCE_SMOKE_NPZ
    rgb_u8 = np.array(
        [
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 255]],
            ],
            [
                [[0, 0, 0], [20, 40, 60]],
                [[80, 100, 120], [140, 160, 180]],
            ],
        ],
        dtype=np.uint8,
    )
    K = np.array(
        [
            [32.0, 0.0, 0.5],
            [0.0, 32.0, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.savez(clip_path, rgb_u8=rgb_u8, K=K)
    return load_npz_clip_frames(clip_path)


def _validate_rgb_clip(field_name: str, value: Array) -> npt.NDArray[np.uint8]:
    rgb = np.asarray(value)
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"{field_name}: must be a uint8 TxHxWx3 array")
    frame_count, height, width, _channel_count = rgb.shape
    if frame_count <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"{field_name}: T, H, and W must be positive")
    return cast(npt.NDArray[np.uint8], rgb)


def _validate_clip_intrinsics(
    field_name: str,
    value: Array,
    frame_count: int,
) -> npt.NDArray[np.float32]:
    intrinsics = validate_finite_numeric_array(field_name, value)
    if intrinsics.shape == (3, 3):
        K = validate_intrinsics(field_name, intrinsics)
        return np.broadcast_to(np.asarray(K, dtype=np.float32), (frame_count, 3, 3)).copy()
    if intrinsics.shape != (frame_count, 3, 3):
        raise ValueError(f"{field_name}: must have shape 3x3 or {frame_count}x3x3")
    for frame_index in range(frame_count):
        validate_intrinsics(f"{field_name}[{frame_index}]", intrinsics[frame_index])
    return np.asarray(intrinsics, dtype=np.float32).copy()


def _frames_from_rgb_clip(
    rgb_u8: npt.NDArray[np.uint8],
    intrinsics: npt.NDArray[np.float32],
    *,
    frame_id_start: int,
    source_format: str,
    source_path: Path,
    source_files: tuple[Path, ...] = (),
) -> tuple[FramePacket, ...]:
    if frame_id_start < 0:
        raise ValueError("frame_id_start: must be non-negative")
    packets: list[FramePacket] = []
    resize_transform = np.eye(3, dtype=np.float32)
    for frame_index, rgb_frame in enumerate(rgb_u8):
        frame_id = frame_id_start + frame_index
        metadata: dict[str, object] = {
            "source_format": source_format,
            "source_path": str(source_path),
            "frame_index": frame_index,
            "timestamp_placeholder": True,
        }
        if source_files:
            metadata["source_file"] = str(source_files[frame_index])
        K = intrinsics[frame_index].astype(np.float32, copy=True)
        packets.append(
            FramePacket(
                frame_id=frame_id,
                timestamp_ns=0,
                rgb_u8=rgb_frame.copy(),
                rgb_model=np.moveaxis(rgb_frame, 2, 0).astype(np.float32, copy=True) / 255.0,
                K_original=K.copy(),
                K_model=K,
                distortion=None,
                resize_transform=resize_transform.copy(),
                camera_metadata=metadata,
            )
        )
    return tuple(packets)


def _load_intrinsics_sidecar(sequence_dir: Path, sidecar: PathLike | None) -> Array:
    sidecar_path = Path(sidecar) if sidecar is not None else sequence_dir / PPM_INTRINSICS_SIDECAR
    if not sidecar_path.is_file():
        raise ValueError(f"{sidecar_path}: missing PPM sequence intrinsics sidecar")
    with np.load(sidecar_path, allow_pickle=False) as payload:
        if "K" not in payload.files:
            raise ValueError(f"{sidecar_path}: missing required K array")
        return np.asarray(payload["K"])


def _read_ppm_p6(path: Path) -> npt.NDArray[np.uint8]:
    data = path.read_bytes()
    magic, index = _read_ppm_token(path, data, 0)
    if magic != b"P6":
        raise ValueError(f"{path}: PPM magic must be P6")
    width_token, index = _read_ppm_token(path, data, index)
    height_token, index = _read_ppm_token(path, data, index)
    max_value_token, index = _read_ppm_token(path, data, index)
    try:
        width = int(width_token)
        height = int(height_token)
        max_value = int(max_value_token)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid PPM header integers") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"{path}: PPM width and height must be positive")
    if max_value != 255:
        raise ValueError(f"{path}: only 8-bit P6 PPM files with max value 255 are supported")
    if index >= len(data) or data[index] not in b" \t\r\n":
        raise ValueError(f"{path}: PPM header must end with whitespace before raster data")
    raster_start = index + 1
    if data[index] == ord("\r") and raster_start < len(data) and data[raster_start] == ord("\n"):
        raster_start += 1
    expected_bytes = width * height * 3
    raster = data[raster_start:]
    if len(raster) != expected_bytes:
        raise ValueError(f"{path}: expected {expected_bytes} raster bytes, got {len(raster)}")
    return np.frombuffer(raster, dtype=np.uint8).reshape((height, width, 3)).copy()


def _read_ppm_token(path: Path, data: bytes, index: int) -> tuple[bytes, int]:
    index = _skip_ppm_whitespace_and_comments(data, index)
    start = index
    while index < len(data) and data[index] not in b" \t\r\n":
        index += 1
    if start == index:
        raise ValueError(f"{path}: truncated PPM header")
    return data[start:index], index


def _skip_ppm_whitespace_and_comments(data: bytes, index: int) -> int:
    while index < len(data):
        if data[index] in b" \t\r\n":
            index += 1
            continue
        if data[index] == ord("#"):
            while index < len(data) and data[index] not in b"\r\n":
                index += 1
            continue
        break
    return index


__all__ = [
    "FRAME_SOURCE_SMOKE_NPZ",
    "PPM_INTRINSICS_SIDECAR",
    "NPZFrameSource",
    "PPMSequenceFrameSource",
    "RGBFrameSource",
    "load_npz_clip_frames",
    "load_ppm_sequence_frames",
    "write_frame_source_smoke_fixture",
]
