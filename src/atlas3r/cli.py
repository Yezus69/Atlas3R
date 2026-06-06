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
from atlas3r.offline.build_world import BuildWorldOptions, build_world
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
    build_world_parser = offline_subparsers.add_parser(
        "build-world", help="Run the connected offline world-builder tracer."
    )
    build_world_parser.add_argument(
        "--input", required=True, help="Input MP4/MOV, PNG/JPG/PPM file, or image folder."
    )
    build_world_parser.add_argument("--output", required=True, help="Output run folder.")
    build_world_parser.add_argument("--max-frames", type=int, default=120)
    build_world_parser.add_argument("--keyframe-stride", type=int, default=5)
    build_world_parser.add_argument("--keyframe-max-count", type=int, default=32)
    build_world_parser.add_argument(
        "--debug-geometry-mode",
        choices=("none", "flat-depth", "synthetic-known"),
        default="none",
    )
    build_world_parser.add_argument("--write-ply", action="store_true")
    build_world_parser.add_argument("--enable-vggt", action="store_true")
    build_world_parser.add_argument("--vggt-repo", default=None)
    build_world_parser.add_argument("--vggt-checkpoint", default=None)
    build_world_parser.add_argument("--vggt-device", default="cuda:0")
    build_world_parser.add_argument("--vggt-image-size", type=int, default=518)
    build_world_parser.add_argument("--vggt-window-size", type=int, default=24)
    build_world_parser.add_argument("--vggt-window-overlap", type=int, default=8)
    build_world_parser.add_argument("--vggt-max-keyframes", type=int, default=None)
    build_world_parser.add_argument("--vggt-proposal-cache", default=None)
    build_world_parser.add_argument(
        "--vggt-stitch-mode",
        choices=("none", "overlap-sim3"),
        default="overlap-sim3",
    )
    build_world_parser.add_argument("--enable-depth-pro", action="store_true")
    build_world_parser.add_argument("--depth-pro-repo", default=None)
    build_world_parser.add_argument("--depth-pro-checkpoint", default=None)
    build_world_parser.add_argument("--depth-pro-device", default="cuda:0")
    build_world_parser.add_argument("--depth-pro-image-size", type=int, default=None)
    build_world_parser.add_argument("--depth-pro-max-keyframes", type=int, default=None)
    build_world_parser.add_argument("--depth-pro-proposal-cache", default=None)
    build_world_parser.set_defaults(func=_run_offline_build_world)

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


def _run_offline_build_world(args: argparse.Namespace) -> int:
    result = build_world(
        BuildWorldOptions(
            input_path=str(args.input),
            output_path=str(args.output),
            max_frames=int(args.max_frames),
            keyframe_stride=int(args.keyframe_stride),
            keyframe_max_count=int(args.keyframe_max_count),
            debug_geometry_mode=args.debug_geometry_mode,
            write_ply=bool(args.write_ply),
            enable_vggt=bool(args.enable_vggt),
            vggt_repo=args.vggt_repo,
            vggt_checkpoint=args.vggt_checkpoint,
            vggt_device=str(args.vggt_device),
            vggt_image_size=int(args.vggt_image_size),
            vggt_window_size=int(args.vggt_window_size),
            vggt_window_overlap=int(args.vggt_window_overlap),
            vggt_max_keyframes=args.vggt_max_keyframes,
            vggt_proposal_cache=args.vggt_proposal_cache,
            vggt_stitch_mode=args.vggt_stitch_mode,
            enable_depth_pro=bool(args.enable_depth_pro),
            depth_pro_repo=args.depth_pro_repo,
            depth_pro_checkpoint=args.depth_pro_checkpoint,
            depth_pro_device=str(args.depth_pro_device),
            depth_pro_image_size=args.depth_pro_image_size,
            depth_pro_max_keyframes=args.depth_pro_max_keyframes,
            depth_pro_proposal_cache=args.depth_pro_proposal_cache,
        )
    )
    print(f"wrote {Path(result.run_dir) / 'run_manifest.json'}")
    print(f"geometry points: {result.geometry_point_count}")
    print(f"failure points: {result.failure_count}")
    return result.exit_code


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
