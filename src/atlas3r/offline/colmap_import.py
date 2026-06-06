"""COLMAP sparse text-model import helpers.

COLMAP image poses are stored as world-to-camera transforms:
``x_camera = R(qvec) * x_world + tvec``. Atlas3R public artifacts use
``T_world_camera``, so imported image poses are inverted before export.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from atlas3r.contracts.coordinates import COORDINATE_FRAME_NAME
from atlas3r.offline.run_manifest import write_jsonl


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "model": self.model,
            "width": self.width,
            "height": self.height,
            "params": list(self.params),
        }


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str
    T_world_camera: NDArray[np.float32]
    camera_center_world_m: NDArray[np.float32]
    point2d_count: int = 0
    observed_point3d_count: int = 0

    @property
    def frame_id(self) -> int | None:
        return frame_id_from_image_name(self.name)

    def to_dict(self) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "frame_id": self.frame_id,
            "name": self.name,
            "qvec_world_to_camera": list(self.qvec),
            "tvec_world_to_camera": list(self.tvec),
            "camera_id": self.camera_id,
            "T_world_camera": self.T_world_camera.tolist(),
            "camera_center_world_m": self.camera_center_world_m.tolist(),
            "point2d_count": self.point2d_count,
            "observed_point3d_count": self.observed_point3d_count,
            "coordinate_convention": COORDINATE_FRAME_NAME,
            "pose_source": "colmap_sfm_unanchored",
        }


@dataclass(frozen=True)
class ColmapPoint3D:
    point3d_id: int
    xyz_world: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float
    track_length: int

    def to_dict(self) -> dict[str, object]:
        return {
            "point3d_id": self.point3d_id,
            "xyz_world": list(self.xyz_world),
            "rgb": list(self.rgb),
            "reprojection_error": self.error,
            "track_length": self.track_length,
        }


@dataclass(frozen=True)
class ColmapSparseModel:
    cameras: tuple[ColmapCamera, ...]
    images: tuple[ColmapImage, ...]
    points3d: tuple[ColmapPoint3D, ...]
    source_path: str

    @property
    def registered_image_count(self) -> int:
        return len(self.images)

    @property
    def sparse_point_count(self) -> int:
        return len(self.points3d)

    def points_xyz_array(self) -> NDArray[np.float32]:
        if not self.points3d:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray([point.xyz_world for point in self.points3d], dtype=np.float32)

    def point_colors_array(self) -> NDArray[np.uint8]:
        if not self.points3d:
            return np.zeros((0, 3), dtype=np.uint8)
        return np.asarray([point.rgb for point in self.points3d], dtype=np.uint8)

    def point_errors_array(self) -> NDArray[np.float32]:
        return np.asarray([point.error for point in self.points3d], dtype=np.float32)

    def track_lengths_array(self) -> NDArray[np.int32]:
        return np.asarray([point.track_length for point in self.points3d], dtype=np.int32)


def read_colmap_text_model(path: str | Path) -> ColmapSparseModel:
    root = Path(path)
    cameras = tuple(_read_cameras(root / "cameras.txt"))
    images = tuple(_read_images(root / "images.txt"))
    points = tuple(_read_points3d(root / "points3D.txt"))
    if not cameras:
        raise ValueError("COLMAP text model contains no cameras")
    if not images:
        raise ValueError("COLMAP text model contains no registered images")
    return ColmapSparseModel(cameras=cameras, images=images, points3d=points, source_path=str(root))


def write_colmap_model_artifacts(
    model: ColmapSparseModel, classical_dir: str | Path
) -> dict[str, str]:
    root = Path(classical_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "colmap_cameras.jsonl", [camera.to_dict() for camera in model.cameras])
    write_jsonl(root / "colmap_images.jsonl", [image.to_dict() for image in model.images])
    np.savez_compressed(
        root / "colmap_points3d.npz",
        points_world_m=model.points_xyz_array(),
        colors_u8=model.point_colors_array(),
        reprojection_error=model.point_errors_array(),
        track_length=model.track_lengths_array(),
        metadata_json=json.dumps(
            {
                "format_name": "atlas3r_colmap_sparse_points",
                "format_version": 1,
                "point_count": model.sparse_point_count,
                "coordinate_convention": COORDINATE_FRAME_NAME,
                "metric_scale_source": "colmap_sfm_unanchored",
            },
            sort_keys=True,
        ),
    )
    write_sparse_points_ply(
        root / "colmap_sparse_points.ply",
        model.points_xyz_array(),
        model.point_colors_array(),
        metric_scale_source="colmap_sfm_unanchored",
    )
    return {
        "colmap_cameras_jsonl": "classical/colmap_cameras.jsonl",
        "colmap_images_jsonl": "classical/colmap_images.jsonl",
        "colmap_points3d_npz": "classical/colmap_points3d.npz",
        "colmap_sparse_points_ply": "classical/colmap_sparse_points.ply",
    }


def write_sparse_points_ply(
    path: str | Path,
    points_world_m: NDArray[np.float32],
    colors_u8: NDArray[np.uint8],
    *,
    metric_scale_source: str,
) -> None:
    truth_boundary = {
        "label_type": "classical_sfm_proposal",
        "measured_geometry": False,
        "observed_only": True,
        "predicted_completion": False,
        "hidden_geometry_measured": False,
        "physical_accuracy_claim": False,
        "scale_status": "sfm_scale_unanchored",
        "metric_scale_source": metric_scale_source,
        "training_quality": False,
        "accuracy_report": False,
        "usable_for_training": False,
    }
    metadata = json.dumps({"truth_boundary": truth_boundary}, sort_keys=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"comment atlas3r_metadata_json {metadata}",
        f"element vertex {int(points_world_m.shape[0])}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "element face 0",
        "property list uchar uint vertex_indices",
        "end_header",
    ]
    for point, color in zip(points_world_m, colors_u8, strict=True):
        lines.append(
            f"{float(point[0]):.8g} {float(point[1]):.8g} {float(point[2]):.8g} "
            f"{int(color[0])} {int(color[1])} {int(color[2])}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def frame_id_from_image_name(name: str) -> int | None:
    stem = Path(name).stem
    if stem.startswith("frame_"):
        suffix = stem.removeprefix("frame_")
        return int(suffix) if suffix.isdigit() else None
    if stem.isdigit():
        return int(stem)
    return None


def qvec_tvec_to_T_world_camera(
    qvec: tuple[float, float, float, float], tvec: tuple[float, float, float]
) -> NDArray[np.float32]:
    rotation_world_to_camera = qvec_to_rotmat(qvec)
    translation = np.asarray(tvec, dtype=np.float32).reshape((3,))
    T_camera_world = np.eye(4, dtype=np.float32)
    T_camera_world[:3, :3] = rotation_world_to_camera
    T_camera_world[:3, 3] = translation
    return np.linalg.inv(T_camera_world).astype(np.float32)


def qvec_to_rotmat(qvec: tuple[float, float, float, float]) -> NDArray[np.float32]:
    qw, qx, qy, qz = (float(value) for value in qvec)
    norm = float(np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("COLMAP quaternion must be finite and nonzero")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
                2.0 * qx * qy - 2.0 * qz * qw,
                2.0 * qx * qz + 2.0 * qy * qw,
            ],
            [
                2.0 * qx * qy + 2.0 * qz * qw,
                1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
                2.0 * qy * qz - 2.0 * qx * qw,
            ],
            [
                2.0 * qx * qz - 2.0 * qy * qw,
                2.0 * qy * qz + 2.0 * qx * qw,
                1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
            ],
        ],
        dtype=np.float32,
    )


def _read_cameras(path: Path) -> list[ColmapCamera]:
    rows = []
    for line_number, line in _data_lines(path):
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"malformed cameras.txt line {line_number}")
        rows.append(
            ColmapCamera(
                camera_id=int(parts[0]),
                model=parts[1],
                width=int(parts[2]),
                height=int(parts[3]),
                params=tuple(float(value) for value in parts[4:]),
            )
        )
    return rows


def _read_images(path: Path) -> list[ColmapImage]:
    data = list(_data_lines(path))
    rows = []
    index = 0
    while index < len(data):
        line_number, image_line = data[index]
        parts = image_line.split()
        if len(parts) < 10:
            raise ValueError(f"malformed images.txt line {line_number}")
        point_line = data[index + 1][1] if index + 1 < len(data) else ""
        point_tokens = point_line.split()
        if len(point_tokens) % 3 != 0:
            raise ValueError(f"malformed images.txt points2D line after {line_number}")
        point3d_ids = [int(value) for value in point_tokens[2::3]]
        qvec = tuple(float(value) for value in parts[1:5])
        if len(qvec) != 4:
            raise ValueError(f"malformed images.txt qvec line {line_number}")
        tvec = tuple(float(value) for value in parts[5:8])
        if len(tvec) != 3:
            raise ValueError(f"malformed images.txt tvec line {line_number}")
        T_world_camera = qvec_tvec_to_T_world_camera(
            (qvec[0], qvec[1], qvec[2], qvec[3]), (tvec[0], tvec[1], tvec[2])
        )
        rows.append(
            ColmapImage(
                image_id=int(parts[0]),
                qvec=(qvec[0], qvec[1], qvec[2], qvec[3]),
                tvec=(tvec[0], tvec[1], tvec[2]),
                camera_id=int(parts[8]),
                name=" ".join(parts[9:]),
                T_world_camera=T_world_camera,
                camera_center_world_m=T_world_camera[:3, 3].astype(np.float32),
                point2d_count=len(point3d_ids),
                observed_point3d_count=sum(1 for point_id in point3d_ids if point_id >= 0),
            )
        )
        index += 2
    return rows


def _read_points3d(path: Path) -> list[ColmapPoint3D]:
    rows = []
    for line_number, line in _data_lines(path):
        parts = line.split()
        if len(parts) < 8:
            raise ValueError(f"malformed points3D.txt line {line_number}")
        track_tokens = parts[8:]
        if len(track_tokens) % 2 != 0:
            raise ValueError(f"malformed points3D.txt track on line {line_number}")
        rows.append(
            ColmapPoint3D(
                point3d_id=int(parts[0]),
                xyz_world=(float(parts[1]), float(parts[2]), float(parts[3])),
                rgb=(int(parts[4]), int(parts[5]), int(parts[6])),
                error=float(parts[7]),
                track_length=len(track_tokens) // 2,
            )
        )
    return rows


def _data_lines(path: Path) -> list[tuple[int, str]]:
    if not path.is_file():
        raise ValueError(f"missing COLMAP text file: {path.name}")
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            rows.append((line_number, stripped))
    return rows
