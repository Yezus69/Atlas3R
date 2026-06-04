"""Runtime CLI registrations for Atlas3R."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def register_runtime_parser(subparsers: Any) -> None:
    runtime_parser = subparsers.add_parser(
        "runtime",
        help="Run Atlas3R runtime diagnostics.",
    )
    runtime_subparsers = runtime_parser.add_subparsers(dest="runtime_command", required=True)
    stream_parser = runtime_subparsers.add_parser(
        "stream-student-map",
        help="Run a temporal checkpoint as a streaming CPU TSDF map diagnostic.",
    )
    stream_parser.add_argument("--checkpoint", type=Path, required=True)
    stream_parser.add_argument("--clip-cache", type=Path, required=True)
    stream_parser.add_argument("--teacher-cache", type=Path, required=True)
    stream_parser.add_argument("--output", type=Path, required=True)
    stream_parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    stream_parser.add_argument("--max-frames", type=int, default=60)
    stream_parser.add_argument("--window-size", type=int, default=5)
    stream_parser.add_argument(
        "--pose-mode",
        choices=("oracle", "student-relative", "student-odometry", "both"),
        default="oracle",
    )
    stream_parser.add_argument("--voxel-size-m", type=float, default=0.05)
    stream_parser.add_argument("--truncation-voxels", type=float, default=3.0)
    stream_parser.set_defaults(handler=_run_stream_student_map)
    fuse_parser = runtime_subparsers.add_parser(
        "fuse-recording",
        help="Fuse measured Atlas3R recording depth+pose through CPU TSDF.",
    )
    fuse_parser.add_argument("--recording", type=Path, required=True)
    fuse_parser.add_argument("--output", type=Path, required=True)
    fuse_parser.add_argument("--pose-source", choices=("recording",), default="recording")
    fuse_parser.add_argument("--depth-source", choices=("recording",), default="recording")
    fuse_parser.add_argument("--max-frames", type=int, default=None)
    fuse_parser.add_argument("--keyframe-stride", type=int, default=1)
    fuse_parser.add_argument("--voxel-size-m", type=float, default=0.05)
    fuse_parser.add_argument("--truncation-voxels", type=float, default=3.0)
    fuse_parser.add_argument("--export-point-cloud", action="store_true")
    fuse_parser.add_argument("--export-mesh", choices=("off", "auto", "required"), default="auto")
    fuse_parser.set_defaults(handler=_run_fuse_recording)


def _run_stream_student_map(args: argparse.Namespace) -> int:
    from atlas3r.runtime.student_map_runtime import (
        StudentMapRuntimeConfig,
        run_stream_student_map,
    )
    from atlas3r.training.torch_runtime import TorchDependencyError

    try:
        result = run_stream_student_map(
            StudentMapRuntimeConfig(
                checkpoint=args.checkpoint,
                clip_cache=args.clip_cache,
                teacher_cache=args.teacher_cache,
                output=args.output,
                device=args.device,
                max_frames=args.max_frames,
                window_size=args.window_size,
                pose_mode=args.pose_mode,
                voxel_size_m=args.voxel_size_m,
                truncation_voxels=args.truncation_voxels,
            )
        )
    except (TorchDependencyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_fuse_recording(args: argparse.Namespace) -> int:
    from atlas3r.runtime.recording_fusion import FuseRecordingConfig, run_fuse_recording

    try:
        result = run_fuse_recording(
            FuseRecordingConfig(
                recording=args.recording,
                output=args.output,
                pose_source=args.pose_source,
                depth_source=args.depth_source,
                max_frames=args.max_frames,
                keyframe_stride=args.keyframe_stride,
                voxel_size_m=args.voxel_size_m,
                truncation_voxels=args.truncation_voxels,
                export_point_cloud=args.export_point_cloud,
                export_mesh=args.export_mesh,
            )
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "register_runtime_parser",
]
