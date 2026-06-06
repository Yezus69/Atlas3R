"""Dependency-safe readers for external ViPE artifact folders."""

from __future__ import annotations

import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


class VipeImportError(RuntimeError):
    """Raised when ViPE artifacts cannot be imported safely."""


@dataclass(frozen=True)
class DepthFrame:
    frame_id: int
    depth_m: NDArray[np.float32]


@dataclass(frozen=True)
class DenseSlamMap:
    points_world_m: NDArray[np.float32]
    colors_u8: NDArray[np.uint8]
    source_frame_ids: NDArray[np.int64]


@dataclass(frozen=True)
class PoseTable:
    frame_ids: tuple[int, ...]
    transforms: tuple[NDArray[np.float32], ...]

    def transform_for(self, frame_id: int, ordinal: int) -> NDArray[np.float32] | None:
        if frame_id in self.frame_ids:
            return self.transforms[self.frame_ids.index(frame_id)]
        if 0 <= ordinal < len(self.transforms):
            return self.transforms[ordinal]
        return None


@dataclass(frozen=True)
class IntrinsicsTable:
    frame_ids: tuple[int, ...]
    intrinsics: tuple[NDArray[np.float32], ...]

    def K_for(self, frame_id: int, ordinal: int) -> NDArray[np.float32] | None:
        if frame_id in self.frame_ids:
            return self.intrinsics[self.frame_ids.index(frame_id)]
        if 0 <= ordinal < len(self.intrinsics):
            return self.intrinsics[ordinal]
        return None


def load_pose_table(vipe_output: Path, artifact_name: str) -> PoseTable:
    path = vipe_output / "pose" / f"{artifact_name}.npz"
    if not path.is_file():
        raise VipeImportError(f"missing ViPE pose artifact: {path}")
    with np.load(path, allow_pickle=False) as payload:
        data = payload["data"].astype(np.float32)
        inds = payload["inds"].astype(np.int64) if "inds" in payload else np.arange(data.shape[0])
    if data.ndim != 3 or data.shape[1:] != (4, 4):
        raise VipeImportError("ViPE pose data must have shape [N,4,4]")
    return PoseTable(
        tuple(int(v) for v in inds.tolist()), tuple(data[i] for i in range(data.shape[0]))
    )


def load_intrinsics_table(vipe_output: Path, artifact_name: str) -> IntrinsicsTable:
    path = vipe_output / "intrinsics" / f"{artifact_name}.npz"
    if not path.is_file():
        raise VipeImportError(f"missing ViPE intrinsics artifact: {path}")
    with np.load(path, allow_pickle=False) as payload:
        data = payload["data"].astype(np.float32)
        inds = payload["inds"].astype(np.int64) if "inds" in payload else np.arange(data.shape[0])
    if data.ndim != 2 or data.shape[1] != 4:
        raise VipeImportError("ViPE intrinsics data must have shape [N,4]")
    intrinsics = tuple(_K_from_vipe_row(row) for row in data)
    return IntrinsicsTable(tuple(int(v) for v in inds.tolist()), intrinsics)


def load_depth_frames(vipe_output: Path, artifact_name: str) -> tuple[DepthFrame, ...]:
    npz_path = vipe_output / "depth" / f"{artifact_name}.npz"
    if npz_path.is_file():
        return _load_depth_npz(npz_path)
    zip_path = vipe_output / "depth" / f"{artifact_name}.zip"
    if zip_path.is_file():
        return _load_depth_zip(zip_path)
    raise VipeImportError(f"missing ViPE depth artifact: {npz_path} or {zip_path}")


def load_slam_dense_map(vipe_output: Path, artifact_name: str) -> DenseSlamMap:
    path = vipe_output / "vipe" / f"{artifact_name}_slam_map.pt"
    if not path.is_file():
        raise VipeImportError(f"missing ViPE dense SLAM map artifact: {path}")
    try:
        import torch
    except ImportError as exc:
        raise VipeImportError("PyTorch is required to import ViPE dense SLAM maps") from exc
    try:
        payload: Any = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise VipeImportError("ViPE dense SLAM map must be a dictionary")
    try:
        xyz_tensor = payload["dense_disp_xyz"]
        rgb_tensor = payload["dense_disp_rgb"]
    except KeyError as exc:
        raise VipeImportError("ViPE dense SLAM map is missing dense point arrays") from exc
    xyz = np.asarray(xyz_tensor.detach().cpu().numpy(), dtype=np.float32)
    rgb = np.asarray(rgb_tensor.detach().cpu().numpy(), dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.all(np.isfinite(xyz)):
        raise VipeImportError("ViPE dense SLAM points must be finite [N,3]")
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise VipeImportError("ViPE dense SLAM colors must have shape [N,3]")
    colors_u8 = _rgb_to_u8(rgb)
    source_frame_ids = _source_frame_ids_from_packinfo(payload, xyz.shape[0])
    return DenseSlamMap(xyz.astype(np.float32), colors_u8, source_frame_ids)


def indexed_frame_paths(frames_dir: Path) -> dict[int, Path]:
    if not frames_dir.is_dir():
        raise VipeImportError(f"missing frame directory: {frames_dir}")
    suffixes = {".jpg", ".jpeg", ".png", ".ppm", ".npy"}
    paths = sorted(path for path in frames_dir.iterdir() if path.suffix.lower() in suffixes)
    return {index: path for index, path in enumerate(paths)}


def load_rgb_for_frame(frame_paths: dict[int, Path], frame_id: int) -> NDArray[np.uint8]:
    path = frame_paths.get(frame_id)
    if path is None:
        raise VipeImportError("source RGB frame is unavailable")
    if path.suffix.lower() == ".ppm":
        return _read_ppm(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.ndim == 3 and array.shape[2] == 3:
            return cast(NDArray[np.uint8], np.clip(array, 0, 255).astype(np.uint8))
        raise VipeImportError(f"RGB npy frame must be H,W,3: {path}")
    try:
        from PIL import Image
    except ImportError as exc:
        raise VipeImportError("Pillow is required to read JPG/PNG source frames") from exc
    with Image.open(path) as image:
        return cast(NDArray[np.uint8], np.asarray(image.convert("RGB"), dtype=np.uint8))


def _load_depth_npz(path: Path) -> tuple[DepthFrame, ...]:
    frames: list[DepthFrame] = []
    with np.load(path, allow_pickle=False) as payload:
        if "depths" in payload:
            depths = payload["depths"].astype(np.float32)
            inds = (
                payload["inds"].astype(np.int64)
                if "inds" in payload
                else np.arange(depths.shape[0])
            )
            return tuple(DepthFrame(int(inds[i]), depths[i]) for i in range(depths.shape[0]))
        for key in sorted(k for k in payload.files if k.startswith("depth_")):
            suffix = key.rsplit("_", 1)[-1]
            frame_id = int(suffix) if suffix.isdigit() else len(frames)
            frames.append(DepthFrame(frame_id, payload[key].astype(np.float32)))
    return tuple(frames)


def _source_frame_ids_from_packinfo(
    payload: dict[str, object], point_count: int
) -> NDArray[np.int64]:
    source = np.full((point_count,), -1, dtype=np.int64)
    frame_inds = payload.get("dense_disp_frame_inds")
    packinfo = payload.get("dense_disp_packinfo")
    if frame_inds is None or packinfo is None:
        return source
    pack = np.asarray(packinfo.cpu().numpy() if hasattr(packinfo, "cpu") else packinfo).reshape(
        (-1, 2)
    )
    frame_list = [int(v) for v in frame_inds] if isinstance(frame_inds, list) else []
    for index, row in enumerate(pack):
        if index >= len(frame_list):
            break
        start = int(row[0])
        count = int(row[1])
        if start >= 0 and count > 0:
            source[start : min(point_count, start + count)] = frame_list[index]
    return source


def _rgb_to_u8(rgb: NDArray[np.float32]) -> NDArray[np.uint8]:
    finite = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    if finite.size and float(finite.max()) <= 1.0:
        finite = finite * 255.0
    return cast(NDArray[np.uint8], np.clip(np.round(finite), 0, 255).astype(np.uint8))


def _load_depth_zip(path: Path) -> tuple[DepthFrame, ...]:
    frames: list[DepthFrame] = []
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if name.lower().endswith(".exr"))
        for name in names:
            stem = Path(name).stem
            frame_id = int(stem) if stem.isdigit() else len(frames)
            frames.append(DepthFrame(frame_id, _read_exr_bytes(archive.read(name))))
    return tuple(frames)


def _read_exr_bytes(payload: bytes) -> NDArray[np.float32]:
    try:
        import Imath  # type: ignore[import-not-found]
        import OpenEXR  # type: ignore[import-not-found]
    except ImportError as exc:
        raise VipeImportError(
            "OpenEXR and Imath are required to import ViPE EXR depth zips"
        ) from exc
    temp_path: Path | None = None
    exr = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as temp:
            temp.write(payload)
            temp_path = Path(temp.name)
        exr = OpenEXR.InputFile(str(temp_path))
        header = exr.header()
        window = header["dataWindow"]
        width = int(window.max.x - window.min.x + 1)
        height = int(window.max.y - window.min.y + 1)
        channels = list(header["channels"].keys())
        channel = "Z" if "Z" in channels else channels[0]
        raw = exr.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
        return np.frombuffer(raw, dtype=np.float32).reshape((height, width)).copy()
    finally:
        if exr is not None:
            exr.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _read_ppm(path: Path) -> NDArray[np.uint8]:
    data = path.read_bytes()
    header, raw = data.split(b"\n255\n", 1)
    parts = header.split()
    if len(parts) != 3 or parts[0] != b"P6":
        raise VipeImportError(f"unsupported PPM frame: {path}")
    width = int(parts[1])
    height = int(parts[2])
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()


def _K_from_vipe_row(row: NDArray[np.float32]) -> NDArray[np.float32]:
    fx, fy, cx, cy = (float(v) for v in row.tolist())
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


__all__ = [
    "DepthFrame",
    "DenseSlamMap",
    "IntrinsicsTable",
    "PoseTable",
    "VipeImportError",
    "indexed_frame_paths",
    "load_depth_frames",
    "load_intrinsics_table",
    "load_pose_table",
    "load_rgb_for_frame",
    "load_slam_dense_map",
]
