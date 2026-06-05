"""Offline teacher witness registry with no heavy import side effects."""

from __future__ import annotations

from atlas3r.teachers.base import AdapterCapabilities, AdapterStatus, UnavailableTeacherAdapter


def list_teacher_statuses() -> tuple[AdapterStatus, ...]:
    return tuple(_STATUSES.values())


def get_teacher_status(name: str) -> AdapterStatus:
    key = _normalize(name)
    if key not in _STATUSES:
        raise KeyError(f"unknown teacher adapter: {name}")
    return _STATUSES[key]


def get_teacher_adapter(name: str) -> UnavailableTeacherAdapter:
    status = get_teacher_status(name)
    return UnavailableTeacherAdapter(name=status.name, status_record=status)


def _normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_")


_STATUSES: dict[str, AdapterStatus] = {
    "depth_pro": AdapterStatus(
        name="depth_pro",
        display_name="Depth Pro",
        available=False,
        capabilities=AdapterCapabilities(depth=True, intrinsics=True, metric_depth=True),
        install_hint=(
            "Install Depth Pro externally and add an adapter under src/atlas3r/models/adapters/."
        ),
        reason="no external model dependency is bundled in this reset foundation",
    ),
    "vggt": AdapterStatus(
        name="vggt",
        display_name="VGGT",
        available=False,
        capabilities=AdapterCapabilities(depth=True, intrinsics=True, pose=True, point_tracks=True),
        install_hint="Install VGGT externally and expose it through a dependency-safe adapter.",
        reason="VGGT is tracked as a future witness, not imported by Atlas3R",
    ),
    "mapanything": AdapterStatus(
        name="mapanything",
        display_name="MapAnything",
        available=False,
        capabilities=AdapterCapabilities(depth=True, pose=True),
        install_hint="Install MapAnything externally before implementing its adapter.",
        reason="adapter is not implemented in the reset foundation",
    ),
    "lingbot_map": AdapterStatus(
        name="lingbot_map",
        display_name="LingBot-Map",
        available=False,
        capabilities=AdapterCapabilities(depth=True, pose=True, streaming=True),
        install_hint="Install LingBot-Map externally before implementing its adapter.",
        reason="adapter is not implemented in the reset foundation",
    ),
    "sam_dino": AdapterStatus(
        name="sam_dino",
        display_name="SAM/DINO",
        available=False,
        capabilities=AdapterCapabilities(object_masks=True),
        install_hint="Install SAM/DINO externally before implementing object proposal adapters.",
        reason="object witness adapter is not implemented yet",
    ),
    "cotracker": AdapterStatus(
        name="cotracker",
        display_name="CoTracker",
        available=False,
        capabilities=AdapterCapabilities(point_tracks=True),
        install_hint="Install CoTracker externally before implementing track proposal adapters.",
        reason="track witness adapter is not implemented yet",
    ),
    "colmap_glomap": AdapterStatus(
        name="colmap_glomap",
        display_name="COLMAP/GLOMAP",
        available=False,
        capabilities=AdapterCapabilities(intrinsics=True, pose=True, point_tracks=True),
        install_hint="Install COLMAP/GLOMAP externally and call it through a process adapter.",
        reason="classical geometry witness is not wired in this reset foundation",
    ),
}
