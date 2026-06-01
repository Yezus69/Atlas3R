"""Dependency-safe VGGT teacher adapter stub."""

from __future__ import annotations

from atlas3r.models.adapters._dependency import build_stub_status, require_optional_modules
from atlas3r.models.adapters.contracts import (
    AdapterCapabilities,
    AdapterNotImplementedError,
    AdapterStatus,
    FrameBatch,
    TeacherPrediction,
)

ADAPTER_NAME = "vggt"
ADAPTER_DISPLAY_NAME = "VGGT"
REQUIRED_MODULES = ("vggt",)
INSTALL_HINT = (
    "Install VGGT in the active environment and keep model weights in an external path; "
    "do not vendor third-party code or weights into Atlas3R."
)
CAPABILITIES = AdapterCapabilities(
    predicts_camera=True,
    predicts_pose=True,
    predicts_depth=True,
    predicts_normals=False,
    predicts_points=True,
    predicts_dense_matches=True,
    predicts_objects=False,
    supports_batch=True,
    supports_streaming=False,
    notes=("Future camera, depth, pointmap, and track teacher.",),
)


def get_adapter_status() -> AdapterStatus:
    return build_stub_status(
        name=ADAPTER_NAME,
        display_name=ADAPTER_DISPLAY_NAME,
        module_names=REQUIRED_MODULES,
        capabilities=CAPABILITIES,
        install_hint=INSTALL_HINT,
    )


class VGGTAdapter:
    name = ADAPTER_NAME

    def __init__(self) -> None:
        require_optional_modules(
            adapter_display_name=ADAPTER_DISPLAY_NAME,
            module_names=REQUIRED_MODULES,
            install_hint=INSTALL_HINT,
        )

    @property
    def status(self) -> AdapterStatus:
        return get_adapter_status()

    def predict(self, frames: FrameBatch) -> TeacherPrediction:
        raise AdapterNotImplementedError(
            "VGGT adapter is stub-only in Phase 0E; dependency checks are wired, "
            "but neural inference is not implemented."
        )


__all__ = ["VGGTAdapter", "get_adapter_status"]
