"""Single-threaded deterministic fixture runtime scheduler skeleton."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atlas3r.data.synthetic_cube_room import write_synthetic_cube_room_session
from atlas3r.io.session import validate_session
from atlas3r.io.teacher_cache import load_teacher_prediction_cache
from atlas3r.mapping.teacher_cache_replay import write_teacher_cache_tsdf_replay
from atlas3r.models.adapters.runner import AdapterRunError, run_adapter_to_cache
from atlas3r.runtime.events import (
    RUNTIME_EVENT_LOG_FORMAT_NAME,
    RUNTIME_EVENT_LOG_FORMAT_VERSION,
    BoundedMemoryCounters,
    RuntimeEvent,
    RuntimeStage,
)

RUNTIME_FIXTURE_SUMMARY_FORMAT_NAME = "atlas3r_runtime_fixture_smoke_summary"
RUNTIME_FIXTURE_SUMMARY_FORMAT_VERSION = 1
RUNTIME_EVENTS_FILENAME = "runtime_events.jsonl"
RUNTIME_SUMMARY_FILENAME = "runtime_summary.json"
DEFAULT_RUNTIME_FRAME_ARRAY_BOUND = 1


@dataclass(frozen=True)
class RuntimeFixtureSmokeResult:
    """In-memory result for `atlas3r smoke runtime-fixture`."""

    output_path: Path
    session_path: Path
    teacher_cache_path: Path
    tsdf_output_path: Path
    event_log_path: Path
    summary_path: Path
    events: tuple[RuntimeEvent, ...]
    summary: dict[str, Any]


class _RuntimeEventRecorder:
    def __init__(self, *, configured_frame_array_bound: int) -> None:
        if configured_frame_array_bound <= 0:
            raise ValueError("configured_frame_array_bound: must be positive")
        self._configured_frame_array_bound = configured_frame_array_bound
        self._events: list[RuntimeEvent] = []
        self._frame_arrays_in_memory = 0
        self._peak_frame_arrays_in_memory = 0
        self._processed_frame_count = 0
        self._dropped_frame_count = 0

    @property
    def configured_frame_array_bound(self) -> int:
        return self._configured_frame_array_bound

    @property
    def events(self) -> tuple[RuntimeEvent, ...]:
        return tuple(self._events)

    @property
    def peak_frame_arrays_in_memory(self) -> int:
        return self._peak_frame_arrays_in_memory

    @property
    def processed_frame_count(self) -> int:
        return self._processed_frame_count

    @property
    def dropped_frame_count(self) -> int:
        return self._dropped_frame_count

    def emit(
        self,
        stage_name: RuntimeStage,
        *,
        frame_id: int | None = None,
        paths: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        dropped_frame: bool = False,
        frame_arrays_in_memory: int = 0,
        count_processed_frame: bool = False,
    ) -> None:
        if frame_arrays_in_memory > self._configured_frame_array_bound:
            raise ValueError(
                "frame_arrays_in_memory: deterministic scheduler bound would be exceeded"
            )
        self._frame_arrays_in_memory = frame_arrays_in_memory
        self._peak_frame_arrays_in_memory = max(
            self._peak_frame_arrays_in_memory,
            self._frame_arrays_in_memory,
        )
        if dropped_frame:
            self._dropped_frame_count += 1
        if count_processed_frame:
            self._processed_frame_count += 1
        event_index = len(self._events)
        counters = BoundedMemoryCounters(
            configured_frame_array_bound=self._configured_frame_array_bound,
            frame_arrays_in_memory=self._frame_arrays_in_memory,
            peak_frame_arrays_in_memory=self._peak_frame_arrays_in_memory,
            processed_frame_count=self._processed_frame_count,
            dropped_frame_count=self._dropped_frame_count,
            queued_frame_count=0,
        )
        self._events.append(
            RuntimeEvent(
                event_index=event_index,
                stage_name=stage_name,
                frame_id=frame_id,
                timestamp_ns=event_index * 1_000_000,
                latency_ns=0,
                dropped_frame=dropped_frame,
                memory_counters=counters,
                paths=paths or {},
                metadata=metadata or {},
            )
        )
        self._frame_arrays_in_memory = 0


def write_runtime_fixture_smoke(
    output_folder: str | Path,
    *,
    frame_array_bound: int = DEFAULT_RUNTIME_FRAME_ARRAY_BOUND,
) -> RuntimeFixtureSmokeResult:
    """Run the deterministic Phase 2D runtime fixture smoke and write artifacts."""
    output_path = Path(output_folder)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"{output_path}: runtime fixture output must be a directory path")
    output_path.mkdir(parents=True, exist_ok=True)

    session_path = output_path / "synthetic_cube_room.atlas3r"
    teacher_cache_path = output_path / "teacher_cache"
    tsdf_output_path = output_path / "teacher_cache_tsdf"
    event_log_path = output_path / RUNTIME_EVENTS_FILENAME
    summary_path = output_path / RUNTIME_SUMMARY_FILENAME
    recorder = _RuntimeEventRecorder(configured_frame_array_bound=frame_array_bound)

    recorder.emit(
        RuntimeStage.RUNTIME_START,
        paths={
            "event_log": RUNTIME_EVENTS_FILENAME,
            "summary": RUNTIME_SUMMARY_FILENAME,
        },
        metadata={
            "scheduler": "runtime-fixture",
            "threads": 0,
            "asyncio": False,
            "deterministic_placeholders": True,
        },
    )

    session_path = write_synthetic_cube_room_session(session_path)
    loaded_session = validate_session(session_path)
    frame_ids = tuple(pose.frame_id for pose in loaded_session.poses)
    recorder.emit(
        RuntimeStage.SESSION_WRITE,
        paths={"session": _rel(output_path, session_path)},
        metadata={
            "session_type": str(loaded_session.metadata["session_type"]),
            "frame_count": len(frame_ids),
        },
    )
    for frame_id in frame_ids:
        recorder.emit(
            RuntimeStage.SOURCE_FRAME,
            frame_id=frame_id,
            frame_arrays_in_memory=1,
            count_processed_frame=True,
            paths={
                "depth_payload": _rel(
                    output_path,
                    session_path / "depth" / f"frame_{frame_id:06d}.npz",
                )
            },
            metadata={"source": "synthetic_cube_room"},
        )

    try:
        run_adapter_to_cache(
            adapter_name="fixture-cube-room",
            input_session=session_path,
            output_cache=teacher_cache_path,
            store_arrays=True,
        )
    except AdapterRunError as exc:
        raise ValueError(f"{teacher_cache_path}: {exc}") from exc
    cache = load_teacher_prediction_cache(teacher_cache_path)
    recorder.emit(
        RuntimeStage.ADAPTER_CACHE_WRITE,
        paths={"teacher_cache": _rel(output_path, teacher_cache_path)},
        metadata={
            "adapter": cache.adapter_status.name,
            "store_arrays": True,
            "frame_count": len(cache.frame_summaries),
        },
    )
    for summary in cache.frame_summaries:
        frame_id = int(summary["frame_id"])
        recorder.emit(
            RuntimeStage.ADAPTER_CACHE_FRAME,
            frame_id=frame_id,
            frame_arrays_in_memory=1,
            paths={
                "array_payload": _rel(
                    output_path,
                    teacher_cache_path / str(summary["arrays_path"]),
                ),
                "teacher_cache": _rel(output_path, teacher_cache_path),
            },
            metadata={"cache_arrays_stored": True},
        )

    recorder.emit(
        RuntimeStage.TSDF_REPLAY,
        paths={
            "teacher_cache": _rel(output_path, teacher_cache_path),
            "tsdf_output": _rel(output_path, tsdf_output_path),
        },
        metadata={
            "mapper_input_contract": "DepthObservation",
            "write_world_map_sidecar": True,
        },
    )
    for summary in cache.frame_summaries:
        frame_id = int(summary["frame_id"])
        recorder.emit(
            RuntimeStage.TSDF_REPLAY_FRAME,
            frame_id=frame_id,
            frame_arrays_in_memory=1,
            paths={
                "array_payload": _rel(
                    output_path,
                    teacher_cache_path / str(summary["arrays_path"]),
                ),
                "tsdf_output": _rel(output_path, tsdf_output_path),
            },
            metadata={"mapper_input_contract": "DepthObservation"},
        )
    try:
        written_tsdf_paths = write_teacher_cache_tsdf_replay(
            teacher_cache_path,
            tsdf_output_path,
            write_world_map_sidecar=True,
        )
    except ValueError as exc:
        raise ValueError(f"{teacher_cache_path}: {exc}") from exc

    recorder.emit(
        RuntimeStage.TSDF_OUTPUT_WRITE,
        paths={
            "tsdf_output": _rel(output_path, tsdf_output_path),
            "metadata": _rel(output_path, tsdf_output_path / "metadata.json"),
            "metrics": _rel(output_path, tsdf_output_path / "metrics.json"),
        },
        metadata={"artifact_count": len(written_tsdf_paths)},
    )
    summary = _summary_record(
        output_path=output_path,
        session_path=session_path,
        teacher_cache_path=teacher_cache_path,
        tsdf_output_path=tsdf_output_path,
        event_log_path=event_log_path,
        frame_ids=frame_ids,
        recorder=recorder,
    )
    recorder.emit(
        RuntimeStage.RUNTIME_COMPLETE,
        paths={
            "event_log": RUNTIME_EVENTS_FILENAME,
            "summary": RUNTIME_SUMMARY_FILENAME,
        },
        metadata={
            "bounded_memory_check_passed": summary["bounded_memory"]["bounded_memory_check_passed"],
            "artifact_count": summary["artifacts"]["artifact_count"],
        },
    )
    summary = _summary_record(
        output_path=output_path,
        session_path=session_path,
        teacher_cache_path=teacher_cache_path,
        tsdf_output_path=tsdf_output_path,
        event_log_path=event_log_path,
        frame_ids=frame_ids,
        recorder=recorder,
    )
    _write_event_log(event_log_path, recorder.events)
    _write_json(summary_path, summary)
    return RuntimeFixtureSmokeResult(
        output_path=output_path,
        session_path=session_path,
        teacher_cache_path=teacher_cache_path,
        tsdf_output_path=tsdf_output_path,
        event_log_path=event_log_path,
        summary_path=summary_path,
        events=recorder.events,
        summary=summary,
    )


def _summary_record(
    *,
    output_path: Path,
    session_path: Path,
    teacher_cache_path: Path,
    tsdf_output_path: Path,
    event_log_path: Path,
    frame_ids: tuple[int, ...],
    recorder: _RuntimeEventRecorder,
) -> dict[str, Any]:
    peak = recorder.peak_frame_arrays_in_memory
    frame_count = len(frame_ids)
    return {
        "artifacts": {
            "artifact_count": len(_artifact_paths(output_path, tsdf_output_path)),
            "event_log": _rel(output_path, event_log_path),
            "paths": _artifact_paths(output_path, tsdf_output_path),
            "runtime_summary": RUNTIME_SUMMARY_FILENAME,
            "session": _rel(output_path, session_path),
            "teacher_cache": _rel(output_path, teacher_cache_path),
            "tsdf_output": _rel(output_path, tsdf_output_path),
        },
        "bounded_memory": {
            "all_frame_arrays_accumulated": peak >= frame_count,
            "bounded_memory_check_passed": peak <= recorder.configured_frame_array_bound,
            "configured_frame_array_bound": recorder.configured_frame_array_bound,
            "counter_scope": "scheduler-owned frame array payload placeholders",
            "dropped_frame_count": recorder.dropped_frame_count,
            "frame_count": frame_count,
            "peak_frame_arrays_in_memory": peak,
            "processed_frame_count": recorder.processed_frame_count,
        },
        "event_log": {
            "event_count": len(recorder.events),
            "format_name": RUNTIME_EVENT_LOG_FORMAT_NAME,
            "format_version": RUNTIME_EVENT_LOG_FORMAT_VERSION,
            "path": _rel(output_path, event_log_path),
        },
        "format_name": RUNTIME_FIXTURE_SUMMARY_FORMAT_NAME,
        "format_version": RUNTIME_FIXTURE_SUMMARY_FORMAT_VERSION,
        "frame_ids": list(frame_ids),
        "runtime": {
            "adapter": "fixture-cube-room",
            "cache_arrays_stored": True,
            "scheduler": "runtime-fixture",
            "single_threaded": True,
            "tsdf_replay_contract": "DepthObservation",
        },
        "truth_boundary": {
            "accuracy_report": False,
            "note": (
                "Runtime fixture smoke uses synthetic analytic cube-room data and is not an "
                "accuracy report."
            ),
        },
    }


def _artifact_paths(output_path: Path, tsdf_output_path: Path) -> list[str]:
    paths = [
        "runtime_events.jsonl",
        "runtime_summary.json",
        "synthetic_cube_room.atlas3r/metadata.json",
        "teacher_cache/metadata.json",
        "teacher_cache/frame_summaries.jsonl",
        "teacher_cache/arrays/frame_000000.npz",
        "teacher_cache/arrays/frame_000001.npz",
        "teacher_cache/arrays/frame_000002.npz",
        _rel(output_path, tsdf_output_path / "tsdf_grid.npz"),
        _rel(output_path, tsdf_output_path / "surface_points.npz"),
        _rel(output_path, tsdf_output_path / "metadata.json"),
        _rel(output_path, tsdf_output_path / "metrics.json"),
        _rel(output_path, tsdf_output_path / "mesh_chunk_sidecar.json"),
        _rel(output_path, tsdf_output_path / "world_map_sidecar.json"),
    ]
    return sorted(paths)


def _write_event_log(path: Path, events: tuple[RuntimeEvent, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event.to_json_record(), sort_keys=True))
            handle.write("\n")


def _write_json(path: Path, record: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


__all__ = [
    "DEFAULT_RUNTIME_FRAME_ARRAY_BOUND",
    "RUNTIME_EVENTS_FILENAME",
    "RUNTIME_FIXTURE_SUMMARY_FORMAT_NAME",
    "RUNTIME_FIXTURE_SUMMARY_FORMAT_VERSION",
    "RUNTIME_SUMMARY_FILENAME",
    "RuntimeFixtureSmokeResult",
    "write_runtime_fixture_smoke",
]
