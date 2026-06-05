"""Torch geometry helpers for SMGT-tiny."""

from __future__ import annotations

from typing import Any

from atlas3r.training.torch_runtime import require_torch

_TORCH: Any = require_torch()
_F: Any = _TORCH.nn.functional


def camera_ray_channels(
    intrinsics: Any,
    *,
    height: int,
    width: int,
    dtype: Any,
    device: Any,
) -> Any:
    """Return `B,3,H,W` ray channels from pinhole intrinsics."""

    batch_size = int(intrinsics.shape[0])
    u = _TORCH.arange(width, dtype=dtype, device=device).view(1, 1, 1, width)
    v = _TORCH.arange(height, dtype=dtype, device=device).view(1, 1, height, 1)
    fx = intrinsics[:, 0, 0].view(batch_size, 1, 1, 1).to(dtype=dtype).clamp(min=1e-6)
    fy = intrinsics[:, 1, 1].view(batch_size, 1, 1, 1).to(dtype=dtype).clamp(min=1e-6)
    cx = intrinsics[:, 0, 2].view(batch_size, 1, 1, 1).to(dtype=dtype)
    cy = intrinsics[:, 1, 2].view(batch_size, 1, 1, 1).to(dtype=dtype)
    ray_x = ((u - cx) / fx).expand(batch_size, 1, height, width)
    ray_y = ((v - cy) / fy).expand(batch_size, 1, height, width)
    radius = _TORCH.sqrt(ray_x.square() + ray_y.square())
    return _TORCH.cat([ray_x, ray_y, radius], dim=1)


def rotation_matrix_from_6d(rotation_6d: Any) -> Any:
    """Convert Zhou-style 6D rotation columns into valid rotation matrices."""

    if int(rotation_6d.shape[-1]) != 6:
        raise ValueError("rotation_6d: last dimension must be 6")
    first = rotation_6d[..., 0:3]
    second = rotation_6d[..., 3:6]
    basis_0 = _F.normalize(first, dim=-1, eps=1e-6)
    second = second - (basis_0 * second).sum(dim=-1, keepdim=True) * basis_0
    basis_1 = _F.normalize(second, dim=-1, eps=1e-6)
    basis_2 = _TORCH.cross(basis_0, basis_1, dim=-1)
    return _TORCH.stack((basis_0, basis_1, basis_2), dim=-1)


def transforms_from_relative_deltas(translation_m: Any, rotation_6d: Any) -> Any:
    """Compose adjacent relative deltas into `B,T,4,4` world-camera transforms."""

    batch_size, frame_count = int(translation_m.shape[0]), int(translation_m.shape[1])
    rotations = rotation_matrix_from_6d(rotation_6d)
    transforms: list[Any] = []
    current = _TORCH.eye(4, dtype=translation_m.dtype, device=translation_m.device)
    current = current.view(1, 4, 4).repeat(batch_size, 1, 1)
    transforms.append(current)
    for index in range(1, frame_count):
        delta = _TORCH.eye(4, dtype=translation_m.dtype, device=translation_m.device)
        delta = delta.view(1, 4, 4).repeat(batch_size, 1, 1)
        delta[:, :3, :3] = rotations[:, index]
        delta[:, :3, 3] = translation_m[:, index]
        current = _TORCH.matmul(current, delta)
        transforms.append(current)
    return _TORCH.stack(transforms, dim=1)


def unproject_depth_camera(depth_m: Any, intrinsics: Any) -> Any:
    """Backproject `B,T,H,W` depth into camera-frame pointmaps `B,T,3,H,W`."""

    batch_size, frame_count, height, width = [int(dim) for dim in depth_m.shape]
    flat_depth = depth_m.reshape(batch_size * frame_count, 1, height, width)
    flat_intrinsics = intrinsics.reshape(batch_size * frame_count, 3, 3)
    rays = camera_ray_channels(
        flat_intrinsics,
        height=height,
        width=width,
        dtype=depth_m.dtype,
        device=depth_m.device,
    )
    pointmap = _TORCH.cat([rays[:, 0:1] * flat_depth, rays[:, 1:2] * flat_depth, flat_depth], dim=1)
    return pointmap.reshape(batch_size, frame_count, 3, height, width)


def normals_from_depth(depth_m: Any) -> Any:
    """Estimate camera-frame normals from local depth gradients."""

    dzdx = _F.pad(depth_m[..., :, 2:] - depth_m[..., :, :-2], (1, 1, 0, 0)) * 0.5
    dzdy = _F.pad(depth_m[..., 2:, :] - depth_m[..., :-2, :], (0, 0, 1, 1)) * 0.5
    normals = _TORCH.stack([-dzdx, -dzdy, _TORCH.ones_like(depth_m)], dim=2)
    return _F.normalize(normals, dim=2, eps=1e-6)


__all__ = [
    "camera_ray_channels",
    "normals_from_depth",
    "rotation_matrix_from_6d",
    "transforms_from_relative_deltas",
    "unproject_depth_camera",
]
