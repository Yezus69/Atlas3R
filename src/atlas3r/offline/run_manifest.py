"""Shared run-manifest and artifact helpers for offline world builds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

RUN_SUBDIRS = (
    "frames",
    "frames/images",
    "keyframes",
    "teachers",
    "proposals",
    "world",
    "geometry",
    "objects",
    "diagnostics",
    "training_cache",
)


@dataclass(frozen=True)
class FailurePoint:
    module: str
    code: str
    severity: str
    status: str
    why: str
    input_missing: str | None = None
    dependency_missing: str | None = None
    future_module: str | None = None
    artifact_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "module": self.module,
            "code": self.code,
            "severity": self.severity,
            "status": self.status,
            "why": self.why,
            "input_missing": self.input_missing,
            "dependency_missing": self.dependency_missing,
            "future_module": self.future_module,
            "artifact_path": self.artifact_path,
        }


def ensure_run_tree(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    for subdir in RUN_SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return root


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def relative_to_run(path: str | Path, run_dir: str | Path) -> str:
    return Path(path).resolve().relative_to(Path(run_dir).resolve()).as_posix()


def write_json(path: str | Path, data: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    target.write_text(payload, encoding="utf-8")
