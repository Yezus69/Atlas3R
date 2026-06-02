"""Deterministic shape-only student model stub."""

from __future__ import annotations

import numpy as np

from atlas3r.models.student.contracts import StudentClipInput, StudentForwardOutput


class ShapeOnlyStudentModel:
    """Return validated placeholder tensors without neural inference."""

    name = "shape-only-student"

    def forward(self, clip_input: StudentClipInput) -> StudentForwardOutput:
        if not isinstance(clip_input, StudentClipInput):
            raise ValueError("clip_input: must be StudentClipInput")

        batch_size = clip_input.batch_size
        frame_count = clip_input.frame_count
        height = clip_input.height
        width = clip_input.width
        bthw_shape = (batch_size, frame_count, height, width)

        depth_m = np.ones(bthw_shape, dtype=np.float32)
        depth_sigma_m = np.ones(bthw_shape, dtype=np.float32)
        confidence = np.zeros(bthw_shape, dtype=np.float32)
        dynamic_probability = np.zeros(bthw_shape, dtype=np.float32)
        normals_camera = np.zeros((batch_size, frame_count, 3, height, width), dtype=np.float32)
        normals_camera[:, :, 2, :, :] = 1.0
        pointmap_camera_m = np.zeros_like(normals_camera)
        T_world_camera = np.broadcast_to(
            np.eye(4, dtype=np.float32),
            (batch_size, frame_count, 4, 4),
        ).copy()
        truth_boundary: dict[str, object] = {
            "shape_only": True,
            "learned_inference": False,
            "usable_for_mapping": False,
            "accuracy_report": False,
            "performance_report": False,
            "source": self.name,
            "coordinate_frame": clip_input.coordinate_frame,
            "note": "Deterministic placeholder tensors only; not neural inference.",
        }

        return StudentForwardOutput(
            frame_ids=clip_input.frame_ids,
            depth_m=depth_m,
            depth_sigma_m=depth_sigma_m,
            confidence=confidence,
            dynamic_probability=dynamic_probability,
            normals_camera=normals_camera,
            pointmap_camera_m=pointmap_camera_m,
            T_world_camera=T_world_camera,
            intrinsics=clip_input.batched_intrinsics(),
            truth_boundary=truth_boundary,
            coordinate_frame=clip_input.coordinate_frame,
        )


__all__ = [
    "ShapeOnlyStudentModel",
]
