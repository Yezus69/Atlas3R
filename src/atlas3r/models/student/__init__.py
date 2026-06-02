"""Dependency-safe student model contracts and shape-only stubs."""

from atlas3r.models.student.contracts import (
    CAMERA_COORDINATE_FRAME,
    STUDENT_TRUTH_BOUNDARY_FLAGS,
    StudentClipInput,
    StudentForwardOutput,
    broadcast_student_intrinsics,
)
from atlas3r.models.student.shape_stub import ShapeOnlyStudentModel

__all__ = [
    "CAMERA_COORDINATE_FRAME",
    "STUDENT_TRUTH_BOUNDARY_FLAGS",
    "ShapeOnlyStudentModel",
    "StudentClipInput",
    "StudentForwardOutput",
    "broadcast_student_intrinsics",
]
