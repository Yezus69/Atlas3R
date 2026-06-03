"""Dependency-safe external teacher runner contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ExternalTeacherError(RuntimeError):
    """Raised when an external teacher runner cannot produce a signal cache."""


class ExternalTeacherDependencyError(ExternalTeacherError):
    """Raised when optional external teacher dependencies are unavailable."""

    def __init__(self, teacher_name: str, reason: str, install_hint: str) -> None:
        super().__init__(f"{teacher_name} is unavailable: {reason}. {install_hint}")
        self.teacher_name = teacher_name
        self.reason = reason
        self.install_hint = install_hint


@dataclass(frozen=True)
class ExternalTeacherStatus:
    name: str
    display_name: str
    availability: str
    can_run_locally: bool
    install_hint: str
    expected_input_format: str
    expected_output_format: str
    reason: str | None = None


@dataclass(frozen=True)
class ExternalTeacherRunConfig:
    clip_cache: Path
    output: Path
    max_clips: int | None = None
    run_inspect: bool = True
    inspect_output: Path | None = None

    def __post_init__(self) -> None:
        if self.max_clips is not None and self.max_clips <= 0:
            raise ValueError("max_clips: must be positive when provided")


class ExternalTeacherRunner(Protocol):
    def status(self) -> ExternalTeacherStatus: ...

    def run(self, config: ExternalTeacherRunConfig) -> dict[str, object]: ...


__all__ = [
    "ExternalTeacherDependencyError",
    "ExternalTeacherError",
    "ExternalTeacherRunConfig",
    "ExternalTeacherRunner",
    "ExternalTeacherStatus",
]
