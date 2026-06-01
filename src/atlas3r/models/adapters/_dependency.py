"""Dependency helpers for optional teacher adapters."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.util import find_spec

from atlas3r.models.adapters.contracts import (
    AdapterAvailability,
    AdapterCapabilities,
    AdapterDependencyError,
    AdapterStatus,
)


def missing_optional_modules(module_names: Sequence[str]) -> tuple[str, ...]:
    missing: list[str] = []
    for module_name in module_names:
        try:
            spec = find_spec(module_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            spec = None
        if spec is None:
            missing.append(module_name)
    return tuple(missing)


def require_optional_modules(
    *,
    adapter_display_name: str,
    module_names: Sequence[str],
    install_hint: str,
) -> None:
    missing = missing_optional_modules(module_names)
    if missing:
        raise AdapterDependencyError(adapter_display_name, missing, install_hint)


def build_stub_status(
    *,
    name: str,
    display_name: str,
    module_names: Sequence[str],
    capabilities: AdapterCapabilities,
    install_hint: str,
) -> AdapterStatus:
    missing = missing_optional_modules(module_names)
    if missing:
        missing_text = ", ".join(missing)
        return AdapterStatus(
            name=name,
            display_name=display_name,
            availability=AdapterAvailability.UNAVAILABLE.value,
            capabilities=capabilities,
            install_hint=install_hint,
            reason=f"missing optional dependency: {missing_text}",
        )
    return AdapterStatus(
        name=name,
        display_name=display_name,
        availability=AdapterAvailability.STUB_ONLY.value,
        capabilities=capabilities,
        install_hint=install_hint,
        reason="optional dependency detected; Phase 0E stub has no inference implementation",
    )
