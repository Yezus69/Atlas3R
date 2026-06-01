"""Tiny teacher adapter runner skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atlas3r.io.session import validate_session
from atlas3r.models.adapters.contracts import AdapterAvailability, AdapterStatus
from atlas3r.models.adapters.registry import get_adapter_status


@dataclass(frozen=True)
class AdapterRunResult:
    adapter_status: AdapterStatus
    input_session: Path
    output_cache: Path


class AdapterRunError(RuntimeError):
    """Raised when an adapter run cannot produce a TeacherPrediction cache."""

    def __init__(
        self,
        *,
        adapter_name: str,
        status: str,
        reason: str,
        guidance: str,
    ) -> None:
        super().__init__(
            "Adapter run failed: "
            f"adapter={adapter_name} status={status} reason={reason} guidance={guidance}"
        )
        self.adapter_name = adapter_name
        self.status = status
        self.reason = reason
        self.guidance = guidance


def run_adapter_to_cache(
    *,
    adapter_name: str,
    input_session: str | Path,
    output_cache: str | Path,
) -> AdapterRunResult:
    """Validate the session and report current adapter run availability."""
    input_path = Path(input_session)
    validate_session(input_path)

    try:
        status = get_adapter_status(adapter_name)
    except KeyError as exc:
        raise AdapterRunError(
            adapter_name=adapter_name,
            status="unknown",
            reason=str(exc),
            guidance="Run `atlas3r adapters list` and choose a known adapter.",
        ) from exc

    if status.availability == AdapterAvailability.UNAVAILABLE.value:
        raise AdapterRunError(
            adapter_name=status.name,
            status=status.availability,
            reason=status.reason or "adapter dependencies are unavailable",
            guidance=status.install_hint or "Install the optional adapter dependencies.",
        )
    if status.availability == AdapterAvailability.STUB_ONLY.value:
        raise AdapterRunError(
            adapter_name=status.name,
            status=status.availability,
            reason=status.reason or "adapter inference is not implemented",
            guidance=(
                "Implement the adapter behind GeometryTeacherAdapter.predict before running; "
                f"{status.install_hint or 'keep third-party code and weights external.'}"
            ),
        )

    raise AdapterRunError(
        adapter_name=status.name,
        status=status.availability,
        reason="Phase 1A runner skeleton does not execute neural inference yet",
        guidance=(
            "Wire the adapter to produce TeacherPrediction, then write it with "
            "write_teacher_prediction_cache."
        ),
    )


__all__ = ["AdapterRunError", "AdapterRunResult", "run_adapter_to_cache"]
