"""SMGT-small-v2 diagnostic student model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from atlas3r.models.smgt.geometry import (
    camera_ray_channels,
    normals_from_depth,
    rotation_matrix_from_6d,
    unproject_depth_camera,
)
from atlas3r.models.smgt.memory import ConvGRUCell
from atlas3r.models.smgt.small_v2_config import SMGTSmallV2Config
from atlas3r.models.student.contracts import StudentForwardOutput
from atlas3r.training.torch_runtime import require_torch

_TORCH: Any = require_torch()
_NN: Any = _TORCH.nn
_F: Any = _TORCH.nn.functional


@dataclass(frozen=True)
class SMGTSmallV2State:
    hidden: Any
    previous_pair_feature: Any
    T_world_camera: Any
    previous_gray: Any | None
    frame_index: int = 0

    def detach(self) -> SMGTSmallV2State:
        return SMGTSmallV2State(
            hidden=self.hidden.detach(),
            previous_pair_feature=self.previous_pair_feature.detach(),
            T_world_camera=self.T_world_camera.detach(),
            previous_gray=None if self.previous_gray is None else self.previous_gray.detach(),
            frame_index=self.frame_index,
        )


class SMGTSmallV2(_NN.Module):  # type: ignore[misc]
    """Ray-aware multi-scale ConvGRU student with pairwise relative pose."""

    def __init__(self, config: SMGTSmallV2Config) -> None:
        super().__init__()
        self.config = config
        s = config.stem_dim
        h = config.hidden_dim
        f = config.feature_dim
        m = config.memory_dim
        self.stem = _convnext_block(6, s, stride=1)
        self.level2 = _convnext_block(s, h, stride=2)
        self.level3 = _convnext_block(h, f, stride=2)
        self.level4 = _convnext_block(f, m, stride=2)
        self.memory = ConvGRUCell(m, m)
        self.up3 = _convnext_block(m + f, f, stride=1)
        self.up2 = _convnext_block(f + h, h, stride=1)
        self.up1 = _convnext_block(h + s, h, stride=1)
        self.log_depth_head = _NN.Conv2d(h, 1, kernel_size=1)
        self.sigma_head = _NN.Conv2d(h, 1, kernel_size=1)
        self.confidence_head = _NN.Conv2d(h, 1, kernel_size=1)
        self.dynamic_head = _NN.Conv2d(h, 1, kernel_size=1)
        self.pose_head = _NN.Sequential(
            _NN.Linear(m * 4 + 3, config.pose_hidden_dim),
            _NN.SiLU(inplace=True),
            _NN.Linear(config.pose_hidden_dim, config.pose_hidden_dim),
            _NN.SiLU(inplace=True),
            _NN.Linear(config.pose_hidden_dim, 10),
        )
        self._initialize_heads()

    def init_state(
        self,
        batch_size: int,
        *,
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> SMGTSmallV2State:
        if batch_size <= 0:
            raise ValueError("batch_size: must be positive")
        device = _TORCH.device("cpu") if device is None else device
        dtype = _TORCH.float32 if dtype is None else dtype
        low_h = (self.config.image_height + 7) // 8
        low_w = (self.config.image_width + 7) // 8
        hidden = _TORCH.zeros(
            (batch_size, self.config.memory_dim, low_h, low_w),
            dtype=dtype,
            device=device,
        )
        pair = _TORCH.zeros((batch_size, self.config.memory_dim), dtype=dtype, device=device)
        transform = _TORCH.eye(4, dtype=dtype, device=device).view(1, 4, 4).repeat(batch_size, 1, 1)
        return SMGTSmallV2State(
            hidden=hidden,
            previous_pair_feature=pair,
            T_world_camera=transform,
            previous_gray=None,
            frame_index=0,
        )

    def forward(self, images_rgb: Any, K: Any) -> dict[str, Any]:
        _validate_clip_inputs(images_rgb, K, self.config)
        batch_size, frame_count = int(images_rgb.shape[0]), int(images_rgb.shape[1])
        state = self.init_state(batch_size, device=images_rgb.device, dtype=images_rgb.dtype)
        outputs: list[dict[str, Any]] = []
        for index in range(frame_count):
            prediction, state = self.step_frame(images_rgb[:, index], K[:, index], state)
            outputs.append(prediction)
        return _cat_frame_outputs(outputs)

    def step_frame(
        self,
        image_rgb: Any,
        K: Any,
        state: SMGTSmallV2State | None = None,
    ) -> tuple[dict[str, Any], SMGTSmallV2State]:
        _validate_step_inputs(image_rgb, K, self.config)
        batch_size = int(image_rgb.shape[0])
        if state is None:
            state = self.init_state(batch_size, device=image_rgb.device, dtype=image_rgb.dtype)
        low, skips, ray_summary = self._encode_frame(image_rgb, K)
        hidden = self.memory(low, state.hidden)
        decoded = self._decode_frame(hidden, skips)
        depth = self._depth_from_logits(self.log_depth_head(decoded))[:, 0].unsqueeze(1)
        sigma = (_F.softplus(self.sigma_head(decoded)) + 1e-4)[:, 0].unsqueeze(1)
        confidence = _TORCH.sigmoid(self.confidence_head(decoded))[:, 0].unsqueeze(1)
        dynamic = _TORCH.sigmoid(self.dynamic_head(decoded))[:, 0].unsqueeze(1)
        gray = _rgb_to_gray(image_rgb)
        phase_translation = self._phase_pose_translation(
            state.previous_gray,
            gray,
            depth,
            K,
            first_frame=state.frame_index == 0,
        )
        pair_feature = hidden.mean(dim=(-2, -1))
        relative_translation, relative_rotation_6d, pose_confidence = self._relative_pose(
            pair_feature,
            state.previous_pair_feature,
            ray_summary,
            phase_translation,
            first_frame=state.frame_index == 0,
        )
        delta = _identity_transform(batch_size, dtype=image_rgb.dtype, device=image_rgb.device)
        delta[:, :3, :3] = rotation_matrix_from_6d(relative_rotation_6d)
        delta[:, :3, 3] = relative_translation
        T_world_camera = _TORCH.matmul(state.T_world_camera.to(image_rgb.device), delta)
        prediction = {
            "depth_m": depth,
            "depth_sigma_m": sigma,
            "confidence": confidence,
            "dynamic_probability": dynamic,
            "normals_camera": normals_from_depth(depth),
            "pointmap_camera_m": unproject_depth_camera(depth, K.unsqueeze(1)),
            "T_world_camera": T_world_camera.unsqueeze(1),
            "intrinsics": K.unsqueeze(1),
            "relative_translation_m": relative_translation.unsqueeze(1),
            "relative_rotation_6d": relative_rotation_6d.unsqueeze(1),
            "pose_confidence": pose_confidence.unsqueeze(1),
        }
        next_state = SMGTSmallV2State(
            hidden=hidden,
            previous_pair_feature=pair_feature,
            T_world_camera=T_world_camera,
            previous_gray=gray.detach(),
            frame_index=state.frame_index + 1,
        )
        return prediction, next_state

    def model_config(self) -> dict[str, object]:
        return {
            "model_name": "SMGTSmallV2",
            "model_role": "diagnostic_smgt_small_v2_measured_pseudo_student",
            "final_smgt": False,
            "object_aware_fusion_implemented": False,
            "input_shape": "B,T,3,H,W RGB plus B,T,3,3 intrinsics-derived ray channels",
            "streaming_api": "init_state(batch_size); step_frame(image, K, state)",
            "outputs": [
                "depth_m B,T,H,W",
                "depth_sigma_m B,T,H,W",
                "confidence B,T,H,W",
                "dynamic_probability B,T,H,W",
                "normals_camera B,T,3,H,W",
                "pointmap_camera_m B,T,3,H,W",
                "relative_translation_m B,T,3",
                "relative_rotation_6d B,T,6",
                "pose_confidence B,T",
                "T_world_camera B,T,4,4",
            ],
            "config": self.config.to_json(),
        }

    def _encode_frame(self, image_rgb: Any, K: Any) -> tuple[Any, tuple[Any, Any, Any], Any]:
        rays = camera_ray_channels(
            K,
            height=int(image_rgb.shape[-2]),
            width=int(image_rgb.shape[-1]),
            dtype=image_rgb.dtype,
            device=image_rgb.device,
        )
        x = _TORCH.cat([image_rgb, rays], dim=1)
        f1 = self.stem(x)
        f2 = self.level2(f1)
        f3 = self.level3(f2)
        f4 = self.level4(f3)
        return f4, (f1, f2, f3), rays.mean(dim=(-2, -1))

    def _decode_frame(self, memory: Any, skips: tuple[Any, Any, Any]) -> Any:
        f1, f2, f3 = skips
        x = _F.interpolate(memory, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(_TORCH.cat([x, f3], dim=1))
        x = _F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(_TORCH.cat([x, f2], dim=1))
        x = _F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        return self.up1(_TORCH.cat([x, f1], dim=1))

    def _depth_from_logits(self, logits: Any) -> Any:
        log_min = _TORCH.log(logits.new_tensor(self.config.min_depth_m))
        log_max = _TORCH.log(logits.new_tensor(self.config.max_depth_m))
        log_depth = log_min + (log_max - log_min) * _TORCH.sigmoid(logits)
        return _TORCH.exp(log_depth).clamp(self.config.min_depth_m, self.config.max_depth_m)

    def _relative_pose(
        self,
        current: Any,
        previous: Any,
        ray_summary: Any,
        phase_translation: Any,
        *,
        first_frame: bool,
    ) -> tuple[Any, Any, Any]:
        pair = _TORCH.cat(
            [current, previous, (current - previous).abs(), current * previous, ray_summary], dim=-1
        )
        raw = self.pose_head(pair)
        learned_translation = (
            self.config.learned_pose_residual_scale * 0.35 * _TORCH.tanh(raw[..., :3])
        )
        translation = phase_translation + learned_translation
        rotation_6d = raw[..., 3:9]
        confidence = _TORCH.sigmoid(raw[..., 9])
        if first_frame:
            translation = _TORCH.zeros_like(translation)
            rotation_6d = _identity_rotation_6d(rotation_6d)
            confidence = _TORCH.ones_like(confidence)
        return translation, rotation_6d, confidence

    def _phase_pose_translation(
        self,
        previous_gray: Any | None,
        current_gray: Any,
        depth_m: Any,
        K: Any,
        *,
        first_frame: bool,
    ) -> Any:
        if first_frame or previous_gray is None or self.config.phase_pose_scale == 0.0:
            return _TORCH.zeros(
                (current_gray.shape[0], 3), dtype=current_gray.dtype, device=current_gray.device
            )
        shift_x, shift_y = _phase_correlation_shift(previous_gray, current_gray)
        max_shift = float(self.config.phase_pose_max_shift_px)
        shift_x = shift_x.clamp(-max_shift, max_shift)
        shift_y = shift_y.clamp(-max_shift, max_shift)
        median_depth = depth_m.detach().float().flatten(2).median(dim=-1).values[:, 0]
        fx = K[:, 0, 0].float().clamp(min=1e-6)
        fy = K[:, 1, 1].float().clamp(min=1e-6)
        scale = float(self.config.phase_pose_scale)
        translation = _TORCH.stack(
            (
                scale * (-shift_x) * median_depth / fx,
                scale * (-shift_y) * median_depth / fy,
                _TORCH.zeros_like(shift_x),
            ),
            dim=-1,
        )
        return translation.to(dtype=current_gray.dtype, device=current_gray.device)

    def _initialize_heads(self) -> None:
        with _TORCH.no_grad():
            self.confidence_head.bias.fill_(-0.5)
            self.dynamic_head.bias.fill_(-3.0)
            final = self.pose_head[-1]
            final.weight.zero_()
            final.bias.zero_()
            final.bias[3:9].copy_(
                _TORCH.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=final.bias.dtype)
            )
            final.bias[9] = 1.0


def smgt_small_v2_prediction_to_student_output(
    prediction: Mapping[str, Any],
    *,
    frame_ids: tuple[int, ...],
    truth_boundary: Mapping[str, object],
) -> StudentForwardOutput:
    return StudentForwardOutput(
        frame_ids=frame_ids,
        depth_m=_numpy(prediction["depth_m"]),
        depth_sigma_m=_numpy(prediction["depth_sigma_m"]),
        confidence=_numpy(prediction["confidence"]),
        dynamic_probability=_numpy(prediction["dynamic_probability"]),
        normals_camera=_numpy(prediction["normals_camera"]),
        pointmap_camera_m=_numpy(prediction["pointmap_camera_m"]),
        T_world_camera=_numpy(prediction["T_world_camera"]),
        intrinsics=_numpy(prediction["intrinsics"]),
        truth_boundary=dict(truth_boundary),
    )


def _convnext_block(in_channels: int, out_channels: int, *, stride: int) -> Any:
    return _NN.Sequential(
        _NN.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        _NN.GroupNorm(_group_count(out_channels), out_channels),
        _NN.SiLU(inplace=True),
        _NN.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, groups=out_channels),
        _NN.GroupNorm(_group_count(out_channels), out_channels),
        _NN.SiLU(inplace=True),
        _NN.Conv2d(out_channels, out_channels, kernel_size=1),
        _NN.SiLU(inplace=True),
    )


def _cat_frame_outputs(outputs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = tuple(outputs[0].keys())
    return {key: _TORCH.cat([output[key] for output in outputs], dim=1) for key in keys}


def _rgb_to_gray(image_rgb: Any) -> Any:
    weights = image_rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (image_rgb * weights).sum(dim=1, keepdim=True)


def _phase_correlation_shift(previous_gray: Any, current_gray: Any) -> tuple[Any, Any]:
    previous = previous_gray.detach().float()
    current = current_gray.detach().float()
    if previous.shape != current.shape:
        raise ValueError("phase pose: previous and current grayscale frames must share shape")
    height, width = int(current.shape[-2]), int(current.shape[-1])
    window_y = _TORCH.hann_window(height, dtype=current.dtype, device=current.device).view(
        1, 1, height, 1
    )
    window_x = _TORCH.hann_window(width, dtype=current.dtype, device=current.device).view(
        1, 1, 1, width
    )
    window = window_y * window_x
    previous = (previous - previous.mean(dim=(-2, -1), keepdim=True)) * window
    current = (current - current.mean(dim=(-2, -1), keepdim=True)) * window
    previous_fft = _TORCH.fft.rfft2(previous[:, 0])
    current_fft = _TORCH.fft.rfft2(current[:, 0])
    cross = current_fft * previous_fft.conj()
    cross = cross / cross.abs().clamp(min=1e-9)
    correlation = _TORCH.fft.irfft2(cross, s=(height, width))
    peak = correlation.flatten(1).argmax(dim=-1)
    shift_y = peak.div(width, rounding_mode="floor").to(dtype=current.dtype)
    shift_x = (peak % width).to(dtype=current.dtype)
    half_width = width // 2
    half_height = height // 2
    shift_x = _TORCH.where(shift_x > half_width, shift_x - width, shift_x)
    shift_y = _TORCH.where(shift_y > half_height, shift_y - height, shift_y)
    return shift_x, shift_y


def _identity_transform(batch_size: int, *, dtype: Any, device: Any) -> Any:
    return _TORCH.eye(4, dtype=dtype, device=device).view(1, 4, 4).repeat(batch_size, 1, 1)


def _identity_rotation_6d(value: Any) -> Any:
    identity = value.new_tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    return identity.view(*((1,) * (value.ndim - 1)), 6).expand_as(value)


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _validate_clip_inputs(images_rgb: Any, K: Any, config: SMGTSmallV2Config) -> None:
    if images_rgb.ndim != 5 or int(images_rgb.shape[2]) != 3:
        raise ValueError("images_rgb: must have shape B,T,3,H,W")
    if K.ndim != 4 or tuple(K.shape[2:]) != (3, 3):
        raise ValueError("K: must have shape B,T,3,3")
    if int(K.shape[0]) != int(images_rgb.shape[0]) or int(K.shape[1]) != int(images_rgb.shape[1]):
        raise ValueError("K: B,T dimensions must match images_rgb")
    if (
        int(images_rgb.shape[-2]) != config.image_height
        or int(images_rgb.shape[-1]) != config.image_width
    ):
        raise ValueError(
            "images_rgb: H,W must match SMGTSmallV2Config "
            f"{config.image_height},{config.image_width}"
        )


def _validate_step_inputs(image_rgb: Any, K: Any, config: SMGTSmallV2Config) -> None:
    if image_rgb.ndim != 4 or int(image_rgb.shape[1]) != 3:
        raise ValueError("image_rgb: must have shape B,3,H,W")
    if K.ndim != 3 or tuple(K.shape[1:]) != (3, 3):
        raise ValueError("K: must have shape B,3,3")
    if int(K.shape[0]) != int(image_rgb.shape[0]):
        raise ValueError("K: B dimension must match image_rgb")
    if (
        int(image_rgb.shape[-2]) != config.image_height
        or int(image_rgb.shape[-1]) != config.image_width
    ):
        raise ValueError(
            "image_rgb: H,W must match SMGTSmallV2Config "
            f"{config.image_height},{config.image_width}"
        )


def _numpy(value: Any) -> np.ndarray[Any, Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


__all__ = [
    "SMGTSmallV2",
    "SMGTSmallV2State",
    "smgt_small_v2_prediction_to_student_output",
]
