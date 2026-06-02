"""Deterministic procedural RGB + metric-depth samples for the training MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.models.student.contracts import CAMERA_COORDINATE_FRAME, StudentClipInput
from atlas3r.pose.transforms import make_transform

ArrayF32 = npt.NDArray[np.float32]
ArrayF64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticDepthSample:
    sample_id: int
    frame_id: int
    rgb_u8: npt.NDArray[np.uint8]
    rgb_model: ArrayF32
    depth_m: ArrayF32
    depth_sigma_m: ArrayF32
    confidence: ArrayF32
    object_mask: npt.NDArray[np.bool_]
    K: ArrayF32
    T_world_camera: ArrayF32
    camera_center_world_m: ArrayF32
    metadata: dict[str, object]


@dataclass(frozen=True)
class _Box:
    min_corner_m: ArrayF64
    max_corner_m: ArrayF64
    color_u8: npt.NDArray[np.uint8]


def generate_synthetic_depth_samples(
    *,
    count: int,
    width: int = 64,
    height: int = 48,
    seed: int = 0,
) -> tuple[SyntheticDepthSample, ...]:
    """Generate deterministic analytic ray-box RGB/depth samples."""

    _validate_positive_int("count", count)
    _validate_positive_int("width", width)
    _validate_positive_int("height", height)
    rng = np.random.default_rng(seed)
    samples: list[SyntheticDepthSample] = []
    for sample_id in range(count):
        sample_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        samples.append(
            _render_sample(
                sample_id=sample_id,
                width=width,
                height=height,
                seed=seed,
                sample_seed=sample_seed,
            )
        )
    return tuple(samples)


def sample_to_student_clip(sample: SyntheticDepthSample) -> StudentClipInput:
    """Convert one synthetic sample into the existing NumPy student boundary."""

    if not isinstance(sample, SyntheticDepthSample):
        raise ValueError("sample: must be a SyntheticDepthSample")
    return StudentClipInput(
        frame_ids=(sample.frame_id,),
        images_rgb=sample.rgb_model[np.newaxis, np.newaxis, :, :, :].astype(
            np.float32,
            copy=True,
        ),
        intrinsics=sample.K[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=True),
        T_world_camera_prior=sample.T_world_camera[np.newaxis, np.newaxis, :, :].astype(
            np.float32,
            copy=True,
        ),
        metadata={
            "source": "procedural_synthetic_depth",
            "sample_id": sample.sample_id,
            "synthetic_only": True,
            "coordinate_frame": CAMERA_COORDINATE_FRAME,
        },
    )


def _render_sample(
    *,
    sample_id: int,
    width: int,
    height: int,
    seed: int,
    sample_seed: int,
) -> SyntheticDepthSample:
    rng = np.random.default_rng(sample_seed)
    room_min = np.array([-1.8, -1.2, 0.0], dtype=np.float64)
    room_max = np.array([1.8, 1.2, float(rng.uniform(3.5, 4.2))], dtype=np.float64)
    camera_center = np.array(
        [
            rng.uniform(-0.35, 0.35),
            rng.uniform(-0.22, 0.22),
            rng.uniform(0.25, 0.6),
        ],
        dtype=np.float64,
    )
    boxes = _make_boxes(rng, room_min, room_max)
    focal = float(max(width, height)) * 0.9
    K = np.array(
        [
            [focal, 0.0, (width - 1.0) * 0.5],
            [0.0, focal, (height - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    T_world_camera = make_transform(np.eye(3, dtype=np.float64), camera_center).astype(np.float32)
    rays_camera = _pixel_rays(width=width, height=height, K=K)
    rgb_flat, depth_flat, object_flat = _trace_scene(
        origin_world=camera_center,
        rays_world=rays_camera,
        room_min=room_min,
        room_max=room_max,
        boxes=boxes,
        rng=rng,
    )
    depth_m = depth_flat.reshape(height, width).astype(np.float32)
    if not np.all(np.isfinite(depth_m)) or np.any(depth_m <= 0.0):
        raise RuntimeError("synthetic depth renderer produced invalid metric depth")
    rgb_u8 = rgb_flat.reshape(height, width, 3).astype(np.uint8)
    rgb_model = np.moveaxis(rgb_u8, 2, 0).astype(np.float32) / np.float32(255.0)
    metadata: dict[str, object] = {
        "synthetic_only": True,
        "seed": seed,
        "sample_seed": sample_seed,
        "scene_bounds_m": {"min": room_min.tolist(), "max": room_max.tolist()},
        "box_count": len(boxes),
        "metric_depth": "analytic_ray_box_intersection",
        "camera_rotation": "identity_looking_down_positive_z",
        "coordinate_frame": CAMERA_COORDINATE_FRAME,
        "limitations": ["axis-aligned boxes", "identity camera rotation", "synthetic-only"],
    }
    return SyntheticDepthSample(
        sample_id=sample_id,
        frame_id=sample_id,
        rgb_u8=rgb_u8,
        rgb_model=rgb_model.astype(np.float32, copy=False),
        depth_m=depth_m,
        depth_sigma_m=np.full((height, width), 0.002, dtype=np.float32),
        confidence=np.ones((height, width), dtype=np.float32),
        object_mask=object_flat.reshape(height, width),
        K=K,
        T_world_camera=T_world_camera,
        camera_center_world_m=camera_center.astype(np.float32),
        metadata=metadata,
    )


def _make_boxes(
    rng: np.random.Generator, room_min: ArrayF64, room_max: ArrayF64
) -> tuple[_Box, ...]:
    box_count = int(rng.integers(1, 4))
    boxes: list[_Box] = []
    palette = np.array(
        [[198, 62, 57], [44, 143, 109], [67, 100, 190], [218, 165, 48]],
        dtype=np.uint8,
    )
    for index in range(box_count):
        if index == 0:
            center_xy = np.array([rng.uniform(-0.35, 0.35), rng.uniform(-0.2, 0.2)])
        else:
            center_xy = np.array([rng.uniform(-0.8, 0.8), rng.uniform(-0.45, 0.45)])
        size = np.array(
            [
                rng.uniform(0.28, 0.62),
                rng.uniform(0.25, 0.58),
                rng.uniform(0.35, 0.9),
            ],
            dtype=np.float64,
        )
        center_z = rng.uniform(1.15, min(3.0, float(room_max[2] - 0.5)))
        center = np.array([center_xy[0], center_xy[1], center_z], dtype=np.float64)
        min_corner = np.maximum(center - size * 0.5, room_min + np.array([0.08, 0.08, 0.7]))
        max_corner = np.minimum(center + size * 0.5, room_max - np.array([0.08, 0.08, 0.08]))
        boxes.append(_Box(min_corner, max_corner, palette[(index + int(rng.integers(0, 4))) % 4]))
    return tuple(boxes)


def _pixel_rays(*, width: int, height: int, K: ArrayF32) -> ArrayF64:
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1).reshape(-1, 3)
    return cast(ArrayF64, rays)


def _trace_scene(
    *,
    origin_world: ArrayF64,
    rays_world: ArrayF64,
    room_min: ArrayF64,
    room_max: ArrayF64,
    boxes: tuple[_Box, ...],
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.uint8], ArrayF64, npt.NDArray[np.bool_]]:
    room_color_a = rng.integers(92, 150, size=3, dtype=np.uint8)
    room_color_b = rng.integers(145, 215, size=3, dtype=np.uint8)
    rgb = np.empty((rays_world.shape[0], 3), dtype=np.uint8)
    depth = np.empty((rays_world.shape[0],), dtype=np.float64)
    object_mask = np.zeros((rays_world.shape[0],), dtype=bool)
    for index, ray_world in enumerate(rays_world):
        best_depth = _intersect_box_exit(origin_world, ray_world, room_min, room_max)
        hit_box: _Box | None = None
        for box in boxes:
            box_depth = _intersect_box_enter(
                origin_world,
                ray_world,
                box.min_corner_m,
                box.max_corner_m,
            )
            if box_depth is not None and box_depth < best_depth:
                best_depth = box_depth
                hit_box = box
        point_world = origin_world + ray_world * best_depth
        depth[index] = best_depth
        if hit_box is None:
            rgb[index] = _room_rgb(point_world, room_min, room_max, room_color_a, room_color_b)
        else:
            rgb[index] = _box_rgb(point_world, hit_box)
            object_mask[index] = True
    return rgb, depth, object_mask


def _room_rgb(
    point_world: ArrayF64,
    room_min: ArrayF64,
    room_max: ArrayF64,
    color_a: npt.NDArray[np.uint8],
    color_b: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    checker = int(np.floor(point_world[0] * 4.0) + np.floor(point_world[1] * 4.0)) & 1
    base = color_a.astype(np.float64) if checker == 0 else color_b.astype(np.float64)
    z_ratio = (point_world[2] - room_min[2]) / max(float(room_max[2] - room_min[2]), 1e-6)
    x_ratio = (point_world[0] - room_min[0]) / max(float(room_max[0] - room_min[0]), 1e-6)
    cue = np.array([40.0 * z_ratio, 35.0 * x_ratio, 30.0 * (1.0 - z_ratio)])
    return cast(npt.NDArray[np.uint8], np.clip(base * 0.82 + cue, 0.0, 255.0).astype(np.uint8))


def _box_rgb(point_world: ArrayF64, box: _Box) -> npt.NDArray[np.uint8]:
    extent = np.maximum(box.max_corner_m - box.min_corner_m, 1e-6)
    local = (point_world - box.min_corner_m) / extent
    stripe = 0.75 + 0.25 * (int(np.floor((local[0] + local[1] + local[2]) * 6.0)) & 1)
    cue = np.array([28.0 * local[0], 24.0 * local[1], 30.0 * local[2]])
    return cast(
        npt.NDArray[np.uint8],
        np.clip(box.color_u8.astype(np.float64) * stripe + cue, 0.0, 255.0).astype(np.uint8),
    )


def _intersect_box_enter(
    origin: ArrayF64,
    direction: ArrayF64,
    min_corner: ArrayF64,
    max_corner: ArrayF64,
) -> float | None:
    enter_depth, exit_depth = _intersect_box(origin, direction, min_corner, max_corner)
    if exit_depth <= 1e-9 or enter_depth <= 1e-9:
        return None
    return enter_depth


def _intersect_box_exit(
    origin: ArrayF64,
    direction: ArrayF64,
    min_corner: ArrayF64,
    max_corner: ArrayF64,
) -> float:
    enter_depth, exit_depth = _intersect_box(origin, direction, min_corner, max_corner)
    if enter_depth > 1e-9:
        return enter_depth
    if exit_depth <= 1e-9:
        raise RuntimeError("camera ray does not hit the synthetic room")
    return exit_depth


def _intersect_box(
    origin: ArrayF64,
    direction: ArrayF64,
    min_corner: ArrayF64,
    max_corner: ArrayF64,
) -> tuple[float, float]:
    enter_depth = -np.inf
    exit_depth = np.inf
    for axis in range(3):
        axis_direction = float(direction[axis])
        axis_origin = float(origin[axis])
        if abs(axis_direction) < 1e-12:
            if axis_origin < min_corner[axis] or axis_origin > max_corner[axis]:
                return np.inf, -np.inf
            continue
        first = (float(min_corner[axis]) - axis_origin) / axis_direction
        second = (float(max_corner[axis]) - axis_origin) / axis_direction
        enter_depth = max(enter_depth, min(first, second))
        exit_depth = min(exit_depth, max(first, second))
        if enter_depth > exit_depth:
            return np.inf, -np.inf
    return float(enter_depth), float(exit_depth)


def _validate_positive_int(field_name: str, value: Any) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name}: must be a positive integer")


__all__ = [
    "SyntheticDepthSample",
    "generate_synthetic_depth_samples",
    "sample_to_student_clip",
]
