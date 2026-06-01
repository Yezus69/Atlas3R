"""Pure-Python discovery for known teacher adapters."""

from __future__ import annotations

from collections.abc import Callable

from atlas3r.models.adapters.contracts import AdapterStatus
from atlas3r.models.adapters.depth_pro_adapter import get_adapter_status as get_depth_pro_status
from atlas3r.models.adapters.fixture_teacher_adapter import (
    get_adapter_status as get_fixture_teacher_status,
)
from atlas3r.models.adapters.vggt_adapter import get_adapter_status as get_vggt_status

StatusFactory = Callable[[], AdapterStatus]

_STATUS_FACTORIES: tuple[StatusFactory, ...] = (
    get_fixture_teacher_status,
    get_vggt_status,
    get_depth_pro_status,
)


def list_adapters() -> tuple[AdapterStatus, ...]:
    return tuple(factory() for factory in _STATUS_FACTORIES)


def get_adapter_status(name: str) -> AdapterStatus:
    for status in list_adapters():
        if status.name == name:
            return status
    known = ", ".join(status.name for status in list_adapters())
    raise KeyError(f"Unknown adapter {name!r}; known adapters: {known}")


__all__ = ["get_adapter_status", "list_adapters"]
