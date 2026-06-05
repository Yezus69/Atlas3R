"""Small Atlas3R command surface for the Offline World Builder reset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from atlas3r import __version__
from atlas3r.contracts.coordinates import COORDINATE_FRAME_NAME
from atlas3r.contracts.frames import CameraModel
from atlas3r.contracts.geometry import PoseEstimate
from atlas3r.contracts.truth import TruthBoundary
from atlas3r.input.video import inspect_video_input
from atlas3r.offline import build_quality_report_skeleton
from atlas3r.teachers.registry import list_teacher_statuses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas3r",
        description="Atlas3R Offline World Builder reset foundation.",
    )
    parser.add_argument("--version", action="version", version=f"atlas3r {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    offline_parser = subparsers.add_parser("offline", help="Offline World Builder commands.")
    offline_subparsers = offline_parser.add_subparsers(dest="offline_command", required=True)
    inspect_video_parser = offline_subparsers.add_parser(
        "inspect-video", help="Inspect an input video or dependency-free PPM sequence."
    )
    inspect_video_parser.add_argument(
        "--input", required=True, help="Input MP4/video path or PPM folder."
    )
    inspect_video_parser.add_argument("--output", required=True, help="Output run folder.")
    inspect_video_parser.set_defaults(func=_run_offline_inspect_video)

    teachers_parser = subparsers.add_parser("teachers", help="Teacher witness registry.")
    teachers_subparsers = teachers_parser.add_subparsers(dest="teachers_command", required=True)
    teachers_list_parser = teachers_subparsers.add_parser("list", help="List teacher availability.")
    teachers_list_parser.add_argument(
        "--json", action="store_true", help="Write machine-readable JSON."
    )
    teachers_list_parser.set_defaults(func=_run_teachers_list)

    smoke_parser = subparsers.add_parser("smoke", help="Small reset smoke checks.")
    smoke_subparsers = smoke_parser.add_subparsers(dest="smoke_command", required=True)
    contracts_parser = smoke_subparsers.add_parser("contracts", help="Validate sample contracts.")
    contracts_parser.set_defaults(func=_run_smoke_contracts)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))


def _run_offline_inspect_video(args: argparse.Namespace) -> int:
    inspection = inspect_video_input(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = build_quality_report_skeleton(inspection, list_teacher_statuses())
    (output / "video_inspection.json").write_text(
        json.dumps(inspection.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "quality_report_skeleton.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output / 'video_inspection.json'}")
    return 1 if inspection.kind == "missing" else 0


def _run_teachers_list(args: argparse.Namespace) -> int:
    statuses = list_teacher_statuses()
    if args.json:
        print(json.dumps([status.to_dict() for status in statuses], indent=2, sort_keys=True))
        return 0
    for status in statuses:
        availability = "available" if status.available else "unavailable"
        print(f"{status.name}\t{availability}\t{status.install_hint}")
    return 0


def _run_smoke_contracts(_args: argparse.Namespace) -> int:
    K = np.array([[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    camera = CameraModel(width=2, height=2, K=K, confidence=1.0, source="synthetic")
    pose = PoseEstimate(
        frame_id=0,
        timestamp_ns=0,
        T_world_camera=np.eye(4, dtype=np.float32),
        covariance_6x6=np.eye(6, dtype=np.float32),
        confidence=1.0,
        tracking_state="OK",
        scale_source="synthetic_gt",
        diagnostics={"coordinate_frame": COORDINATE_FRAME_NAME},
    )
    truth = TruthBoundary(label_type="synthetic_gt", metric_scale_source="synthetic_gt")
    camera.to_dict()
    pose.to_dict()
    truth.to_dict()
    print("contracts ok")
    return 0
