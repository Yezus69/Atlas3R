"""Incremental backend selector for measured recording TSDF fusion."""

from __future__ import annotations

from atlas3r.runtime.recording_fusion import FuseRecordingConfig
from atlas3r.runtime.recording_fusion_incremental_helpers import (
    CPU_REBUILD_BACKEND,
    _resolved_backend,
)
from atlas3r.runtime.recording_fusion_incremental_persistent import (
    run_cpu_persistent_incremental,
)
from atlas3r.runtime.recording_fusion_incremental_rebuild import run_cpu_rebuild_incremental


class IncrementalRecordingFusion:
    """Run selected measured keyframes through the requested incremental backend."""

    def __init__(self, config: FuseRecordingConfig) -> None:
        self._config = config

    def run(self) -> dict[str, object]:
        backend = _resolved_backend(self._config)
        if backend == CPU_REBUILD_BACKEND:
            return run_cpu_rebuild_incremental(self._config, backend)
        return run_cpu_persistent_incremental(self._config, backend)


__all__ = ["IncrementalRecordingFusion"]
