"""Teacher adapter boundary and dependency-safe registry."""

from atlas3r.teachers.base import (
    AdapterCapabilities,
    AdapterStatus,
    GeometryTeacherAdapter,
    TeacherUnavailableError,
    UnavailableTeacherAdapter,
)
from atlas3r.teachers.registry import get_teacher_adapter, get_teacher_status, list_teacher_statuses

__all__ = [
    "AdapterCapabilities",
    "AdapterStatus",
    "GeometryTeacherAdapter",
    "TeacherUnavailableError",
    "UnavailableTeacherAdapter",
    "get_teacher_adapter",
    "get_teacher_status",
    "list_teacher_statuses",
]
