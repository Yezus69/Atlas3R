"""Object permanence ledger skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atlas3r.offline.proposal_cache import ProposalCacheResult
from atlas3r.offline.run_manifest import FailurePoint, write_json


@dataclass(frozen=True)
class ObjectLedgerResult:
    status: str
    object_ledger_path: str
    object_count: int


def write_object_ledger(
    run_dir: str | Path,
    *,
    proposal_cache: ProposalCacheResult,
    failure_points: list[FailurePoint],
) -> ObjectLedgerResult:
    sam_streams = [
        stream for stream in proposal_cache.streams if stream["teacher_name"] == "sam_dino"
    ]
    status = "unavailable"
    why = "SAM/DINO mask and feature proposals are unavailable; no objects are invented."
    failure_points.append(
        FailurePoint(
            module="object_permanence_ledger",
            code="object_witness_missing",
            severity="warning",
            status=status,
            why=why,
            input_missing="SAM/DINO masks or object features",
            future_module="object mask and feature proposal adapter",
            artifact_path="objects/object_ledger.json",
        )
    )
    payload = {
        "status": status,
        "why": why,
        "object_count": 0,
        "objects": [],
        "static_scene_separation": "not_available",
        "proposal_sources": sam_streams,
        "invented_objects": False,
    }
    write_json(Path(run_dir) / "objects" / "object_ledger.json", payload)
    return ObjectLedgerResult(
        status=status, object_ledger_path="objects/object_ledger.json", object_count=0
    )
