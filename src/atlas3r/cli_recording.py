"""Recording CLI registrations for Atlas3R."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from atlas3r.recording.importers import (
    ClipCacheRecordingImportConfig,
    SensorFolderRecordingImportConfig,
    TumRecordingImportConfig,
    recording_from_clip_cache,
    recording_from_sensor_folder,
    recording_from_tum_manifest,
)
from atlas3r.recording.schema import validate_recording_folder


def register_recording_parser(subparsers: Any) -> None:
    recording_parser = subparsers.add_parser(
        "recording",
        help="Validate and import Atlas3R recording streams.",
    )
    recording_subparsers = recording_parser.add_subparsers(
        dest="recording_command",
        required=True,
    )
    validate_parser = recording_subparsers.add_parser(
        "validate",
        help="Validate an Atlas3R recording folder and print deterministic JSON.",
    )
    validate_parser.add_argument("--input", type=Path, required=True)
    validate_parser.set_defaults(handler=_run_validate)

    tum_parser = recording_subparsers.add_parser(
        "from-tum",
        help="Import a TUM RGB-D manifest into an Atlas3R recording.",
    )
    tum_parser.add_argument("--manifest", type=Path, required=True)
    tum_parser.add_argument("--output", type=Path, required=True)
    tum_parser.add_argument("--split", choices=("train", "val"), default="val")
    tum_parser.add_argument("--max-frames", type=int, default=None)
    tum_parser.add_argument("--width", type=int, required=True)
    tum_parser.add_argument("--height", type=int, required=True)
    tum_parser.set_defaults(handler=_run_from_tum)

    clip_parser = recording_subparsers.add_parser(
        "from-clip-cache",
        help="Import an Atlas3R clip cache into a recording folder.",
    )
    clip_parser.add_argument("--clip-cache", type=Path, required=True)
    clip_parser.add_argument("--output", type=Path, required=True)
    clip_parser.add_argument("--dedupe-frame-id", action="store_true")
    clip_parser.set_defaults(handler=_run_from_clip_cache)

    sensor_parser = recording_subparsers.add_parser(
        "from-sensor-folder",
        help="Import a generic measured RGB-D/pose sensor folder into a recording.",
    )
    sensor_parser.add_argument("--input", type=Path, required=True)
    sensor_parser.add_argument("--output", type=Path, required=True)
    sensor_parser.set_defaults(handler=_run_from_sensor_folder)


def _run_validate(args: argparse.Namespace) -> int:
    try:
        result = validate_recording_folder(args.input)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_from_tum(args: argparse.Namespace) -> int:
    try:
        result = recording_from_tum_manifest(
            TumRecordingImportConfig(
                manifest=args.manifest,
                output=args.output,
                split=args.split,
                max_frames=args.max_frames,
                width=args.width,
                height=args.height,
            )
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_from_clip_cache(args: argparse.Namespace) -> int:
    try:
        result = recording_from_clip_cache(
            ClipCacheRecordingImportConfig(
                clip_cache=args.clip_cache,
                output=args.output,
                dedupe_frame_id=args.dedupe_frame_id,
            )
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_from_sensor_folder(args: argparse.Namespace) -> int:
    try:
        result = recording_from_sensor_folder(
            SensorFolderRecordingImportConfig(
                input=args.input,
                output=args.output,
            )
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "register_recording_parser",
]
