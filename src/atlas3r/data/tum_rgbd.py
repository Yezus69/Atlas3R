"""Minimal TUM RGB-D ingestion and manifest preparation."""

from __future__ import annotations

import json
import math
import shutil
import tarfile
import urllib.error
import urllib.request
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

TUM_RGBD_MANIFEST_FORMAT = "atlas3r_tum_rgbd_manifest"
TUM_RGBD_MANIFEST_VERSION = 1
TUM_RGBD_COORDINATE_FRAME = "x_right_y_down_z_forward"

_FREIBURG1_XYZ_ARCHIVE = "rgbd_dataset_freiburg1_xyz.tgz"
_FREIBURG1_XYZ_GROUNDTRUTH = "rgbd_dataset_freiburg1_xyz-groundtruth.txt"

TUM_RGBD_SEQUENCES: dict[str, dict[str, object]] = {
    "freiburg1_xyz": {
        "sequence_name": "freiburg1_xyz",
        "dataset_name": "TUM RGB-D freiburg1_xyz",
        "archive_name": _FREIBURG1_XYZ_ARCHIVE,
        "archive_url": (
            "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/"
            "rgbd_dataset_freiburg1_xyz.tgz"
        ),
        "groundtruth_name": _FREIBURG1_XYZ_GROUNDTRUTH,
        "groundtruth_url": (
            "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/"
            "rgbd_dataset_freiburg1_xyz-groundtruth.txt"
        ),
        "extracted_dir": "rgbd_dataset_freiburg1_xyz",
        "width": 640,
        "height": 480,
        "intrinsics": {
            "fx": 525.0,
            "fy": 525.0,
            "cx": 319.5,
            "cy": 239.5,
        },
        "depth_scale": 5000.0,
    }
}

_T = TypeVar("_T")


@dataclass(frozen=True)
class TumImageEntry:
    timestamp_s: float
    relative_path: str


@dataclass(frozen=True)
class TumPoseEntry:
    timestamp_s: float
    tx_m: float
    ty_m: float
    tz_m: float
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass(frozen=True)
class TumAssociatedFrame:
    rgb: TumImageEntry
    depth: TumImageEntry
    pose: TumPoseEntry


def download_tum_rgbd_sequence(sequence: str, output: str | Path) -> tuple[Path, ...]:
    """Download and safely extract the supported TUM RGB-D sequence files."""

    spec = _sequence_spec(sequence)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    archive_path = output_path / str(spec["archive_name"])
    groundtruth_path = output_path / str(spec["groundtruth_name"])
    _download_file(str(spec["archive_url"]), archive_path)
    _download_file(str(spec["groundtruth_url"]), groundtruth_path)
    safe_extract_tar(archive_path, output_path)
    extracted = output_path / str(spec["extracted_dir"])
    if not extracted.is_dir():
        raise ValueError(f"{archive_path}: expected extracted folder {extracted}")
    shutil.copyfile(groundtruth_path, extracted / "groundtruth.txt")
    return archive_path, groundtruth_path, extracted


def safe_extract_tar(archive: str | Path, output_dir: str | Path) -> None:
    """Extract a tar archive after rejecting unsafe member paths."""

    archive_path = Path(archive)
    target_root = Path(output_dir).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, mode="r:*") as tar:
            members = tar.getmembers()
            for member in members:
                _validate_tar_member(archive_path, target_root, member)
            tar.extractall(target_root, members=members)
    except tarfile.TarError as exc:
        raise ValueError(f"{archive_path}: failed to extract tar archive: {exc}") from exc


def prepare_tum_rgbd_manifest(
    input_dir: str | Path,
    output: str | Path,
    *,
    sequence: str = "freiburg1_xyz",
    stride: int = 1,
    max_frames: int | None = None,
    max_delta_s: float = 0.02,
) -> dict[str, object]:
    """Prepare a deterministic Atlas3R manifest from a TUM RGB-D sequence folder."""

    _validate_prepare_args(stride=stride, max_frames=max_frames, max_delta_s=max_delta_s)
    spec = _sequence_spec(sequence)
    root = Path(input_dir)
    rgb_entries = parse_tum_image_list(root / "rgb.txt")
    depth_entries = parse_tum_image_list(root / "depth.txt")
    pose_entries = parse_tum_groundtruth(root / "groundtruth.txt")
    associated = associate_tum_rgbd_frames(
        rgb_entries,
        depth_entries,
        pose_entries,
        max_delta_s=max_delta_s,
    )
    selected = associated[::stride]
    if max_frames is not None:
        selected = selected[:max_frames]
    if not selected:
        raise ValueError(f"{root}: no associated RGB/depth/pose frames after stride/max-frames")

    records = [
        _frame_record(root, frame_id=frame_id, associated_frame=frame, split=_split(frame_id))
        for frame_id, frame in enumerate(selected)
    ]
    train_count = sum(1 for record in records if record["split"] == "train")
    val_count = sum(1 for record in records if record["split"] == "val")
    manifest = {
        "format_name": TUM_RGBD_MANIFEST_FORMAT,
        "format_version": TUM_RGBD_MANIFEST_VERSION,
        "dataset_name": str(spec["dataset_name"]),
        "sequence_name": str(spec["sequence_name"]),
        "root_path": str(root),
        "coordinate_frame": TUM_RGBD_COORDINATE_FRAME,
        "image": {
            "width": _int_spec(spec, "width"),
            "height": _int_spec(spec, "height"),
            "rgb_semantics": "8-bit RGB PNG",
            "depth_semantics": "16-bit monochrome PNG, pre-registered to RGB",
        },
        "intrinsics": {
            "K": _intrinsics_matrix(spec),
            "source": "TUM RGB-D ROS default freiburg1 intrinsics",
        },
        "depth": {
            "scale": _float_spec(spec, "depth_scale"),
            "meters_formula": "depth_raw / 5000.0",
            "zero_is_missing": True,
            "valid_depth_mask": "depth_raw > 0",
        },
        "association": {
            "max_delta_s": max_delta_s,
            "stride": stride,
            "max_frames": max_frames,
            "rgb_depth_pose_timestamp_source": "nearest within max_delta_s",
        },
        "split": {
            "policy": "deterministic: every tenth associated frame is validation",
            "train_count": train_count,
            "val_count": val_count,
        },
        "frame_count": len(records),
        "frames": records,
        "truth_boundary": _truth_boundary(),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def load_tum_rgbd_manifest(path: str | Path) -> dict[str, object]:
    """Load and minimally validate a TUM RGB-D manifest."""

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"{manifest_path}: failed to read manifest: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path}: invalid JSON manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path}: manifest must be a JSON object")
    if payload.get("format_name") != TUM_RGBD_MANIFEST_FORMAT:
        raise ValueError(f"{manifest_path}: expected format_name={TUM_RGBD_MANIFEST_FORMAT!r}")
    if int(payload.get("format_version", -1)) != TUM_RGBD_MANIFEST_VERSION:
        raise ValueError(f"{manifest_path}: unsupported format_version")
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError(f"{manifest_path}: frames must be a non-empty list")
    return dict(payload)


def parse_tum_image_list(path: str | Path) -> tuple[TumImageEntry, ...]:
    """Parse `rgb.txt` or `depth.txt` style timestamp/path records."""

    rows: list[TumImageEntry] = []
    list_path = Path(path)
    for line_number, line in _iter_data_lines(list_path):
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{list_path}:{line_number}: expected '<timestamp> <relative_path>'")
        try:
            timestamp_s = float(parts[0])
        except ValueError as exc:
            raise ValueError(f"{list_path}:{line_number}: invalid timestamp") from exc
        rows.append(TumImageEntry(timestamp_s=timestamp_s, relative_path=parts[1]))
    if not rows:
        raise ValueError(f"{list_path}: no image records found")
    return tuple(sorted(rows, key=lambda row: (row.timestamp_s, row.relative_path)))


def parse_tum_groundtruth(path: str | Path) -> tuple[TumPoseEntry, ...]:
    """Parse TUM ground-truth trajectory records."""

    rows: list[TumPoseEntry] = []
    gt_path = Path(path)
    for line_number, line in _iter_data_lines(gt_path):
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(
                f"{gt_path}:{line_number}: expected timestamp tx ty tz qx qy qz qw"
            )
        try:
            values = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"{gt_path}:{line_number}: invalid numeric pose value") from exc
        rows.append(TumPoseEntry(*values))
    if not rows:
        raise ValueError(f"{gt_path}: no ground-truth pose records found")
    return tuple(sorted(rows, key=lambda row: row.timestamp_s))


def associate_tum_rgbd_frames(
    rgb_entries: tuple[TumImageEntry, ...],
    depth_entries: tuple[TumImageEntry, ...],
    pose_entries: tuple[TumPoseEntry, ...],
    *,
    max_delta_s: float = 0.02,
) -> tuple[TumAssociatedFrame, ...]:
    """Associate RGB, depth, and pose rows by nearest timestamp."""

    if max_delta_s <= 0.0:
        raise ValueError("max_delta_s: must be positive")
    depth_timestamps = tuple(entry.timestamp_s for entry in depth_entries)
    pose_timestamps = tuple(entry.timestamp_s for entry in pose_entries)
    associated: list[TumAssociatedFrame] = []
    for rgb in rgb_entries:
        depth = _nearest_by_timestamp(depth_entries, depth_timestamps, rgb.timestamp_s)
        pose = _nearest_by_timestamp(pose_entries, pose_timestamps, rgb.timestamp_s)
        if depth is None or pose is None:
            continue
        if abs(depth.timestamp_s - rgb.timestamp_s) > max_delta_s:
            continue
        if abs(pose.timestamp_s - rgb.timestamp_s) > max_delta_s:
            continue
        associated.append(TumAssociatedFrame(rgb=rgb, depth=depth, pose=pose))
    if not associated:
        raise ValueError("TUM RGB-D association produced no frames within max_delta_s")
    return tuple(associated)


def _download_file(url: str, output: Path) -> None:
    tmp = output.with_suffix(output.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except (OSError, urllib.error.URLError) as exc:
        if tmp.exists():
            tmp.unlink()
        raise ValueError(f"{url}: failed to download to {output}: {exc}") from exc
    tmp.replace(output)


def _validate_tar_member(archive_path: Path, target_root: Path, member: tarfile.TarInfo) -> None:
    member_path = Path(member.name)
    if member_path.is_absolute():
        raise ValueError(f"{archive_path}: unsafe absolute tar member {member.name!r}")
    if ".." in member_path.parts:
        raise ValueError(f"{archive_path}: unsafe parent traversal tar member {member.name!r}")
    if member.issym() or member.islnk():
        raise ValueError(f"{archive_path}: unsafe link tar member {member.name!r}")
    destination = (target_root / member.name).resolve()
    if not destination.is_relative_to(target_root):
        raise ValueError(f"{archive_path}: tar member escapes output directory {member.name!r}")


def _iter_data_lines(path: Path) -> tuple[tuple[int, str], ...]:
    if not path.is_file():
        raise ValueError(f"{path}: required TUM RGB-D file does not exist")
    rows: list[tuple[int, str]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                rows.append((line_number, line))
    except OSError as exc:
        raise ValueError(f"{path}: failed to read TUM RGB-D file: {exc}") from exc
    return tuple(rows)


def _nearest_by_timestamp(
    entries: tuple[_T, ...],
    timestamps: tuple[float, ...],
    timestamp_s: float,
) -> _T | None:
    if not entries:
        return None
    index = bisect_left(timestamps, timestamp_s)
    candidates: list[int] = []
    if index < len(entries):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    if not candidates:
        return None
    best_index = min(candidates, key=lambda candidate: abs(timestamps[candidate] - timestamp_s))
    return entries[best_index]


def _frame_record(
    root: Path,
    *,
    frame_id: int,
    associated_frame: TumAssociatedFrame,
    split: str,
) -> dict[str, object]:
    pose = associated_frame.pose
    transform = _T_world_camera_from_pose(pose)
    rgb_path = root / associated_frame.rgb.relative_path
    depth_path = root / associated_frame.depth.relative_path
    if not rgb_path.is_file():
        raise ValueError(f"{rgb_path}: associated RGB PNG does not exist")
    if not depth_path.is_file():
        raise ValueError(f"{depth_path}: associated depth PNG does not exist")
    return {
        "frame_id": frame_id,
        "split": split,
        "rgb_timestamp_s": associated_frame.rgb.timestamp_s,
        "depth_timestamp_s": associated_frame.depth.timestamp_s,
        "pose_timestamp_s": pose.timestamp_s,
        "rgb_path": associated_frame.rgb.relative_path,
        "depth_path": associated_frame.depth.relative_path,
        "T_world_camera": transform,
        "q_world_camera_xyzw": [pose.qx, pose.qy, pose.qz, pose.qw],
        "camera_center_world_m": [pose.tx_m, pose.ty_m, pose.tz_m],
        "pose_source": "TUM RGB-D groundtruth.txt",
    }


def _T_world_camera_from_pose(pose: TumPoseEntry) -> list[list[float]]:
    rotation = _rotation_matrix_from_quaternion_xyzw(pose.qx, pose.qy, pose.qz, pose.qw)
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], pose.tx_m],
        [rotation[1][0], rotation[1][1], rotation[1][2], pose.ty_m],
        [rotation[2][0], rotation[2][1], rotation[2][2], pose.tz_m],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rotation_matrix_from_quaternion_xyzw(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> list[list[float]]:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        raise ValueError("groundtruth.txt: quaternion norm must be positive")
    x = qx / norm
    y = qy / norm
    z = qz / norm
    w = qw / norm
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _intrinsics_matrix(spec: dict[str, object]) -> list[list[float]]:
    values = spec["intrinsics"]
    if not isinstance(values, dict):
        raise ValueError("sequence intrinsics spec must be a dictionary")
    return [
        [float(values["fx"]), 0.0, float(values["cx"])],
        [0.0, float(values["fy"]), float(values["cy"])],
        [0.0, 0.0, 1.0],
    ]


def _int_spec(spec: dict[str, object], field_name: str) -> int:
    value = spec.get(field_name)
    if not isinstance(value, int):
        raise ValueError(f"sequence spec {field_name}: must be an integer")
    return value


def _float_spec(spec: dict[str, object], field_name: str) -> float:
    value = spec.get(field_name)
    if not isinstance(value, int | float):
        raise ValueError(f"sequence spec {field_name}: must be numeric")
    return float(value)


def _sequence_spec(sequence: str) -> dict[str, object]:
    if sequence not in TUM_RGBD_SEQUENCES:
        supported = ", ".join(sorted(TUM_RGBD_SEQUENCES))
        raise ValueError(f"sequence: unsupported TUM RGB-D sequence {sequence!r}; supported: {supported}")
    return TUM_RGBD_SEQUENCES[sequence]


def _validate_prepare_args(
    *,
    stride: int,
    max_frames: int | None,
    max_delta_s: float,
) -> None:
    if stride <= 0:
        raise ValueError("stride: must be a positive integer")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames: must be positive when provided")
    if max_delta_s <= 0.0:
        raise ValueError("max_delta_s: must be positive")


def _split(frame_id: int) -> str:
    return "val" if frame_id % 10 == 0 else "train"


def _truth_boundary() -> dict[str, object]:
    return {
        "training_mvp": True,
        "trained_on_real_rgbd": True,
        "dataset": "TUM RGB-D freiburg1_xyz",
        "synthetic_only": False,
        "real_capture_debug_model": True,
        "usable_for_realtime_mapping": False,
        "usable_for_mapping": False,
        "accuracy_report": False,
        "performance_report": False,
        "learned_inference": True,
        "generalizes_to_real_world": False,
    }


__all__ = [
    "TUM_RGBD_COORDINATE_FRAME",
    "TUM_RGBD_MANIFEST_FORMAT",
    "TUM_RGBD_MANIFEST_VERSION",
    "TumAssociatedFrame",
    "TumImageEntry",
    "TumPoseEntry",
    "associate_tum_rgbd_frames",
    "download_tum_rgbd_sequence",
    "load_tum_rgbd_manifest",
    "parse_tum_groundtruth",
    "parse_tum_image_list",
    "prepare_tum_rgbd_manifest",
    "safe_extract_tar",
]
