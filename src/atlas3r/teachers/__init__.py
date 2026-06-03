"""Teacher-signal cache contracts and diagnostics."""

from atlas3r.teachers.map_eval import (
    TeacherSignalInspectConfig,
    TeacherSignalMapConfig,
    inspect_teacher_signals,
    map_teacher_signals,
)
from atlas3r.teachers.measured_tum import (
    LocalTeacherIngestConfig,
    MeasuredTumTeacherForgeConfig,
    forge_measured_tum_teacher_signal_cache,
    ingest_local_teacher_signal_cache,
)
from atlas3r.teachers.signals import (
    TEACHER_SIGNAL_FORMAT_NAME,
    TEACHER_SIGNAL_FORMAT_VERSION,
    TEACHER_SIGNAL_MANIFEST_FILENAME,
    load_teacher_signal_manifest,
    read_teacher_signal_payload,
    validate_teacher_signal_manifest,
    validate_teacher_signal_payload,
    write_teacher_signal_payload,
)

__all__ = [
    "LocalTeacherIngestConfig",
    "MeasuredTumTeacherForgeConfig",
    "TEACHER_SIGNAL_FORMAT_NAME",
    "TEACHER_SIGNAL_FORMAT_VERSION",
    "TEACHER_SIGNAL_MANIFEST_FILENAME",
    "TeacherSignalInspectConfig",
    "TeacherSignalMapConfig",
    "forge_measured_tum_teacher_signal_cache",
    "ingest_local_teacher_signal_cache",
    "inspect_teacher_signals",
    "load_teacher_signal_manifest",
    "map_teacher_signals",
    "read_teacher_signal_payload",
    "validate_teacher_signal_manifest",
    "validate_teacher_signal_payload",
    "write_teacher_signal_payload",
]
