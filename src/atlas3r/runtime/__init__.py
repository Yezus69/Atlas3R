"""Live camera, video IO, and runtime scheduling."""

from atlas3r.runtime.events import (
    RUNTIME_EVENT_LOG_FORMAT_NAME,
    RUNTIME_EVENT_LOG_FORMAT_VERSION,
    BoundedMemoryCounters,
    RuntimeEvent,
    RuntimeStage,
)
from atlas3r.runtime.scheduler import (
    DEFAULT_RUNTIME_FRAME_ARRAY_BOUND,
    RUNTIME_EVENTS_FILENAME,
    RUNTIME_FIXTURE_SUMMARY_FORMAT_NAME,
    RUNTIME_FIXTURE_SUMMARY_FORMAT_VERSION,
    RUNTIME_SUMMARY_FILENAME,
    RuntimeFixtureSmokeResult,
    write_runtime_fixture_smoke,
)

__all__ = [
    "BoundedMemoryCounters",
    "DEFAULT_RUNTIME_FRAME_ARRAY_BOUND",
    "RUNTIME_EVENTS_FILENAME",
    "RUNTIME_EVENT_LOG_FORMAT_NAME",
    "RUNTIME_EVENT_LOG_FORMAT_VERSION",
    "RUNTIME_FIXTURE_SUMMARY_FORMAT_NAME",
    "RUNTIME_FIXTURE_SUMMARY_FORMAT_VERSION",
    "RUNTIME_SUMMARY_FILENAME",
    "RuntimeEvent",
    "RuntimeFixtureSmokeResult",
    "RuntimeStage",
    "write_runtime_fixture_smoke",
]
