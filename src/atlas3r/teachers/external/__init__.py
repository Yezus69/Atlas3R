"""Dependency-isolated external teacher runners."""

from atlas3r.teachers.external.contracts import (
    ExternalTeacherDependencyError,
    ExternalTeacherError,
    ExternalTeacherRunConfig,
    ExternalTeacherRunner,
    ExternalTeacherStatus,
)
from atlas3r.teachers.external.depth_pro import (
    DepthProExternalTeacherRunner,
    DepthProFramePrediction,
    DepthProRunConfig,
    get_depth_pro_status,
    run_depth_pro_teacher_signal_cache,
)
from atlas3r.teachers.external.vggt_local import (
    VGGTLocalIngestConfig,
    get_vggt_local_status,
    ingest_vggt_local_teacher_signal_cache,
)

__all__ = [
    "DepthProExternalTeacherRunner",
    "DepthProFramePrediction",
    "DepthProRunConfig",
    "ExternalTeacherDependencyError",
    "ExternalTeacherError",
    "ExternalTeacherRunConfig",
    "ExternalTeacherRunner",
    "ExternalTeacherStatus",
    "VGGTLocalIngestConfig",
    "get_depth_pro_status",
    "get_vggt_local_status",
    "ingest_vggt_local_teacher_signal_cache",
    "run_depth_pro_teacher_signal_cache",
]
