"""Third-party teacher model adapter contracts and discovery."""

from atlas3r.models.adapters.contracts import (
    AdapterAvailability,
    AdapterCapabilities,
    AdapterDependencyError,
    AdapterNotImplementedError,
    AdapterStatus,
    FrameBatch,
    GeometryTeacherAdapter,
    TeacherPrediction,
)
from atlas3r.models.adapters.depth_pro_adapter import DepthProAdapter
from atlas3r.models.adapters.registry import get_adapter_status, list_adapters
from atlas3r.models.adapters.runner import AdapterRunError, AdapterRunResult, run_adapter_to_cache
from atlas3r.models.adapters.vggt_adapter import VGGTAdapter

__all__ = [
    "AdapterAvailability",
    "AdapterCapabilities",
    "AdapterDependencyError",
    "AdapterNotImplementedError",
    "AdapterRunError",
    "AdapterRunResult",
    "AdapterStatus",
    "DepthProAdapter",
    "FrameBatch",
    "GeometryTeacherAdapter",
    "TeacherPrediction",
    "VGGTAdapter",
    "get_adapter_status",
    "list_adapters",
    "run_adapter_to_cache",
]
