"""Validation helpers for runtime fixture output inspection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from atlas3r.runtime.events import (
    RUNTIME_EVENT_LOG_FORMAT_NAME,
    RUNTIME_EVENT_LOG_FORMAT_VERSION,
    RuntimeStage,
)
from atlas3r.runtime.scheduler import (
    RUNTIME_EVENTS_FILENAME,
    RUNTIME_FIXTURE_SUMMARY_FORMAT_NAME,
    RUNTIME_FIXTURE_SUMMARY_FORMAT_VERSION,
)

_FRAME_STAGES = {
    RuntimeStage.SOURCE_FRAME.value,
    RuntimeStage.ADAPTER_CACHE_FRAME.value,
    RuntimeStage.TSDF_REPLAY_FRAME.value,
}


@dataclass(frozen=True)
class EventValidation:
    stage_sequence: list[str]
    stage_counts: dict[str, int]
    frame_ids_by_stage: dict[str, list[int]]
    relative_paths: list[str]
    configured_frame_array_bound: int
    peak_frame_arrays_in_memory: int
    processed_frame_count: int
    dropped_frame_count: int
    queued_frame_count: int


def read_json_object(path: Path, *, missing_kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{path}: missing required {missing_kind}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return cast(dict[str, Any], data)


def read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise ValueError(f"{path}: missing required runtime event log")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(cast(dict[str, Any], data))
    return tuple(records)


def validate_summary_header(summary: dict[str, Any], path: Path) -> list[int]:
    if required(summary, "format_name", path) != RUNTIME_FIXTURE_SUMMARY_FORMAT_NAME:
        raise ValueError(f"{path}: format_name must be {RUNTIME_FIXTURE_SUMMARY_FORMAT_NAME}")
    if int(required(summary, "format_version", path)) != RUNTIME_FIXTURE_SUMMARY_FORMAT_VERSION:
        raise ValueError(f"{path}: format_version must be {RUNTIME_FIXTURE_SUMMARY_FORMAT_VERSION}")
    frame_ids = int_list_field(summary, "frame_ids", path)
    if sorted(frame_ids) != frame_ids:
        raise ValueError(f"{path}: frame_ids must be sorted")
    return frame_ids


def validate_events(
    root: Path,
    event_log_path: Path,
    events: tuple[dict[str, Any], ...],
    frame_ids: list[int],
) -> EventValidation:
    if not events:
        raise ValueError(f"{event_log_path}: expected at least one runtime event")
    expected_sequence, expected_frame_ids_by_stage = _expected_event_layout(frame_ids)
    stage_sequence: list[str] = []
    frame_ids_by_stage: dict[str, list[int]] = {stage: [] for stage in sorted(_FRAME_STAGES)}
    stage_counts = {stage.value: 0 for stage in RuntimeStage}
    relative_paths: list[str] = []
    configured_bound: int | None = None
    peak_frame_arrays_in_memory = 0
    processed_frame_count = 0
    dropped_frame_count = 0
    queued_frame_count = 0
    previous_processed = 0
    previous_dropped = 0
    previous_peak = 0

    for index, event in enumerate(events):
        location = Path(f"{event_log_path}:{index + 1}")
        if required(event, "format_name", location) != RUNTIME_EVENT_LOG_FORMAT_NAME:
            raise ValueError(f"{location}: format_name must be {RUNTIME_EVENT_LOG_FORMAT_NAME}")
        if int(required(event, "format_version", location)) != RUNTIME_EVENT_LOG_FORMAT_VERSION:
            raise ValueError(
                f"{location}: format_version must be {RUNTIME_EVENT_LOG_FORMAT_VERSION}"
            )
        event_index = int_field(event, "event_index", location)
        if event_index != index:
            raise ValueError(f"{location}: event_index must be {index}")
        stage_name = str_field(event, "stage_name", location)
        if stage_name not in stage_counts:
            raise ValueError(f"{location}: unknown stage_name {stage_name!r}")
        stage_sequence.append(stage_name)
        stage_counts[stage_name] += 1
        if int_field(event, "timestamp_ns", location) != index * 1_000_000:
            raise ValueError(f"{location}: timestamp_ns must be deterministic placeholder")
        if int_field(event, "latency_ns", location) != 0:
            raise ValueError(f"{location}: latency_ns must be deterministic placeholder 0")
        if bool_field(event, "dropped_frame", location):
            raise ValueError(f"{location}: runtime fixture should not mark dropped frames")
        frame_id = frame_id_field(event, location)
        if stage_name in _FRAME_STAGES:
            if frame_id is None:
                raise ValueError(f"{location}: frame_id is required for {stage_name}")
            frame_ids_by_stage[stage_name].append(frame_id)
        elif frame_id is not None:
            raise ValueError(f"{location}: frame_id must be null for {stage_name}")
        relative_paths.extend(_validate_paths(root, dict_field(event, "paths", location), location))
        dict_field(event, "metadata", location)
        counters = _validate_memory_counters(
            dict_field(event, "memory_counters", location), location
        )
        if configured_bound is None:
            configured_bound = counters["configured_frame_array_bound"]
        elif configured_bound != counters["configured_frame_array_bound"]:
            raise ValueError(f"{location}: configured_frame_array_bound changed")
        if counters["processed_frame_count"] < previous_processed:
            raise ValueError(f"{location}: processed_frame_count must be monotonic")
        if counters["dropped_frame_count"] < previous_dropped:
            raise ValueError(f"{location}: dropped_frame_count must be monotonic")
        if counters["peak_frame_arrays_in_memory"] < previous_peak:
            raise ValueError(f"{location}: peak_frame_arrays_in_memory must be monotonic")
        previous_processed = counters["processed_frame_count"]
        previous_dropped = counters["dropped_frame_count"]
        previous_peak = counters["peak_frame_arrays_in_memory"]
        peak_frame_arrays_in_memory = max(
            peak_frame_arrays_in_memory,
            counters["peak_frame_arrays_in_memory"],
        )
        processed_frame_count = counters["processed_frame_count"]
        dropped_frame_count = counters["dropped_frame_count"]
        queued_frame_count = counters["queued_frame_count"]

    if stage_sequence != expected_sequence:
        raise ValueError(f"{event_log_path}: event stage sequence does not match runtime fixture")
    for stage_name, expected_frame_ids in expected_frame_ids_by_stage.items():
        if frame_ids_by_stage[stage_name] != expected_frame_ids:
            raise ValueError(f"{event_log_path}: frame_id ordering mismatch for {stage_name}")
    if configured_bound is None:
        raise ValueError(f"{event_log_path}: missing bounded-memory counters")
    return EventValidation(
        stage_sequence=stage_sequence,
        stage_counts=stage_counts,
        frame_ids_by_stage=frame_ids_by_stage,
        relative_paths=sorted(set(relative_paths)),
        configured_frame_array_bound=configured_bound,
        peak_frame_arrays_in_memory=peak_frame_arrays_in_memory,
        processed_frame_count=processed_frame_count,
        dropped_frame_count=dropped_frame_count,
        queued_frame_count=queued_frame_count,
    )


def validate_summary(
    summary: dict[str, Any],
    path: Path,
    event_validation: EventValidation,
    frame_ids: list[int],
) -> dict[str, Any]:
    event_log = dict_field(summary, "event_log", path)
    require_equal(
        required(event_log, "format_name", path),
        RUNTIME_EVENT_LOG_FORMAT_NAME,
        path,
        "event_log.format_name",
        RUNTIME_EVENT_LOG_FORMAT_NAME,
    )
    require_equal(
        int(required(event_log, "format_version", path)),
        RUNTIME_EVENT_LOG_FORMAT_VERSION,
        path,
        "event_log.format_version",
        str(RUNTIME_EVENT_LOG_FORMAT_VERSION),
    )
    require_equal(
        required(event_log, "path", path),
        RUNTIME_EVENTS_FILENAME,
        path,
        "event_log.path",
        RUNTIME_EVENTS_FILENAME,
    )
    require_equal(
        int(required(event_log, "event_count", path)),
        len(event_validation.stage_sequence),
        path,
        "event_log.event_count",
        "runtime_events.jsonl",
    )
    bounded_memory = dict_field(summary, "bounded_memory", path)
    _validate_summary_bounded_memory(bounded_memory, path, event_validation, frame_ids)
    runtime = dict_field(summary, "runtime", path)
    require_equal(
        required(runtime, "adapter", path),
        "fixture-cube-room",
        path,
        "runtime.adapter",
        "fixture-cube-room",
    )
    require_equal(
        bool_field(runtime, "cache_arrays_stored", path),
        True,
        path,
        "runtime.cache_arrays_stored",
        "true",
    )
    require_equal(
        required(runtime, "tsdf_replay_contract", path),
        "DepthObservation",
        path,
        "runtime.tsdf_replay_contract",
        "DepthObservation",
    )
    truth = dict_field(summary, "truth_boundary", path)
    require_false(
        bool_field(truth, "accuracy_report", path), path, "truth_boundary.accuracy_report"
    )
    if "not an accuracy report" not in str_field(truth, "note", path):
        raise ValueError(f"{path}: truth_boundary.note must state not an accuracy report")
    return {
        "format_name": summary["format_name"],
        "format_version": summary["format_version"],
        "frame_ids": frame_ids,
        "event_count": event_log["event_count"],
        "runtime": {
            "adapter": runtime["adapter"],
            "cache_arrays_stored": runtime["cache_arrays_stored"],
            "scheduler": runtime["scheduler"],
            "single_threaded": runtime["single_threaded"],
            "tsdf_replay_contract": runtime["tsdf_replay_contract"],
        },
        "bounded_memory": {
            "configured_frame_array_bound": bounded_memory["configured_frame_array_bound"],
            "peak_frame_arrays_in_memory": bounded_memory["peak_frame_arrays_in_memory"],
            "all_frame_arrays_accumulated": bounded_memory["all_frame_arrays_accumulated"],
            "bounded_memory_check_passed": bounded_memory["bounded_memory_check_passed"],
            "processed_frame_count": bounded_memory["processed_frame_count"],
            "dropped_frame_count": bounded_memory["dropped_frame_count"],
        },
        "truth_boundary": truth,
    }


def validate_relative_path(root: Path, value: str, path: Path, field_name: str) -> str:
    if not value:
        raise ValueError(f"{path}: {field_name} must be non-empty")
    if "\\" in value:
        raise ValueError(f"{path}: {field_name} must use forward-slash relative paths")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{path}: {field_name} must be relative to the runtime fixture folder")
    if not (root / relative).exists():
        raise ValueError(f"{path}: {field_name} points to missing path {value}")
    return value


def required(record: Mapping[str, Any], field_name: str, path: Path) -> Any:
    if field_name not in record:
        raise ValueError(f"{path}: missing field {field_name}")
    return record[field_name]


def dict_field(record: Mapping[str, Any], field_name: str, path: Path) -> dict[str, Any]:
    value = required(record, field_name, path)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {field_name} must be an object")
    return cast(dict[str, Any], value)


def str_field(record: Mapping[str, Any], field_name: str, path: Path) -> str:
    value = required(record, field_name, path)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: {field_name} must be a non-empty string")
    return value


def bool_field(record: Mapping[str, Any], field_name: str, path: Path) -> bool:
    value = required(record, field_name, path)
    if not isinstance(value, bool):
        raise ValueError(f"{path}: {field_name} must be a bool")
    return value


def int_field(record: Mapping[str, Any], field_name: str, path: Path) -> int:
    value = required(record, field_name, path)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path}: {field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{path}: {field_name} must be non-negative")
    return value


def frame_id_field(record: Mapping[str, Any], path: Path) -> int | None:
    value = required(record, "frame_id", path)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path}: frame_id must be a non-negative integer or null")
    return value


def int_list_field(record: Mapping[str, Any], field_name: str, path: Path) -> list[int]:
    value = required(record, field_name, path)
    if not isinstance(value, list):
        raise ValueError(f"{path}: {field_name} must be a list")
    result: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ValueError(f"{path}: {field_name}[{index}] must be a non-negative integer")
        result.append(item)
    return result


def string_list_field(record: Mapping[str, Any], field_name: str, path: Path) -> list[str]:
    value = required(record, field_name, path)
    if not isinstance(value, list):
        raise ValueError(f"{path}: {field_name} must be a list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{path}: {field_name} must contain non-empty strings")
    return cast(list[str], value)


def require_equal(left: Any, right: Any, path: Path, field_name: str, source: str) -> None:
    if left != right:
        raise ValueError(f"{path}: {field_name} mismatch with {source}")


def require_false(value: bool, path: Path, field_name: str) -> None:
    if value:
        raise ValueError(f"{path}: {field_name} must be false")


def _expected_event_layout(frame_ids: list[int]) -> tuple[list[str], dict[str, list[int]]]:
    sequence = [
        RuntimeStage.RUNTIME_START.value,
        RuntimeStage.SESSION_WRITE.value,
        *[RuntimeStage.SOURCE_FRAME.value for _ in frame_ids],
        RuntimeStage.ADAPTER_CACHE_WRITE.value,
        *[RuntimeStage.ADAPTER_CACHE_FRAME.value for _ in frame_ids],
        RuntimeStage.TSDF_REPLAY.value,
        *[RuntimeStage.TSDF_REPLAY_FRAME.value for _ in frame_ids],
        RuntimeStage.TSDF_OUTPUT_WRITE.value,
        RuntimeStage.RUNTIME_COMPLETE.value,
    ]
    return sequence, {
        RuntimeStage.SOURCE_FRAME.value: list(frame_ids),
        RuntimeStage.ADAPTER_CACHE_FRAME.value: list(frame_ids),
        RuntimeStage.TSDF_REPLAY_FRAME.value: list(frame_ids),
    }


def _validate_paths(root: Path, paths: dict[str, Any], path: Path) -> list[str]:
    relative_paths: list[str] = []
    for key, value in paths.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{path}: paths keys must be non-empty strings")
        if not isinstance(value, str):
            raise ValueError(f"{path}: paths.{key} must be a string")
        relative_paths.append(validate_relative_path(root, value, path, f"paths.{key}"))
    return relative_paths


def _validate_memory_counters(record: dict[str, Any], path: Path) -> dict[str, int]:
    counters = {
        "configured_frame_array_bound": int_field(record, "configured_frame_array_bound", path),
        "dropped_frame_count": int_field(record, "dropped_frame_count", path),
        "frame_arrays_in_memory": int_field(record, "frame_arrays_in_memory", path),
        "peak_frame_arrays_in_memory": int_field(record, "peak_frame_arrays_in_memory", path),
        "processed_frame_count": int_field(record, "processed_frame_count", path),
        "queued_frame_count": int_field(record, "queued_frame_count", path),
    }
    if counters["configured_frame_array_bound"] <= 0:
        raise ValueError(f"{path}: configured_frame_array_bound must be positive")
    for field_name, value in counters.items():
        if value < 0:
            raise ValueError(f"{path}: {field_name} must be non-negative")
    if counters["frame_arrays_in_memory"] > counters["configured_frame_array_bound"]:
        raise ValueError(f"{path}: frame_arrays_in_memory exceeds configured bound")
    if counters["peak_frame_arrays_in_memory"] > counters["configured_frame_array_bound"]:
        raise ValueError(f"{path}: peak_frame_arrays_in_memory exceeds configured bound")
    return counters


def _validate_summary_bounded_memory(
    record: dict[str, Any],
    path: Path,
    event_validation: EventValidation,
    frame_ids: list[int],
) -> None:
    checks = {
        "configured_frame_array_bound": event_validation.configured_frame_array_bound,
        "peak_frame_arrays_in_memory": event_validation.peak_frame_arrays_in_memory,
        "processed_frame_count": event_validation.processed_frame_count,
        "dropped_frame_count": event_validation.dropped_frame_count,
        "frame_count": len(frame_ids),
    }
    for field_name, expected in checks.items():
        require_equal(
            int_field(record, field_name, path),
            expected,
            path,
            f"bounded_memory.{field_name}",
            "runtime_events.jsonl",
        )
    require_equal(
        bool_field(record, "all_frame_arrays_accumulated", path),
        event_validation.peak_frame_arrays_in_memory >= len(frame_ids),
        path,
        "bounded_memory.all_frame_arrays_accumulated",
        "bounded memory counters",
    )
    require_equal(
        bool_field(record, "bounded_memory_check_passed", path),
        True,
        path,
        "bounded_memory.bounded_memory_check_passed",
        "true",
    )


__all__ = [
    "EventValidation",
    "bool_field",
    "dict_field",
    "read_json_object",
    "read_jsonl",
    "require_equal",
    "require_false",
    "required",
    "str_field",
    "string_list_field",
    "validate_events",
    "validate_relative_path",
    "validate_summary",
    "validate_summary_header",
]
