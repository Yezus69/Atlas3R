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
    fuse_parser.add_argument("--mode", choices=("batch", "incremental"), default="batch")
    fuse_parser.add_argument(
        "--backend",
        choices=("cpu-persistent", "cpu-rebuild", "cpu-sparse"),
        default=None,
    )
    fuse_parser.set_defaults(handler=_run_fuse_recording)
    replay_parser = runtime_subparsers.add_parser(
        "live-replay-recording",
        help="Replay a measured Atlas3R recording through bounded live-style queues.",
    )
    replay_parser.add_argument("--recording", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    replay_parser.add_argument("--target-fps", type=float, default=30.0)
    replay_parser.add_argument("--max-frames", type=int, default=None)
    replay_parser.add_argument(
        "--mapper-backend",
        choices=("cpu-sparse",),
        default="cpu-sparse",
    )
    replay_parser.add_argument("--map-keyframe-stride", type=int, default=1)
    replay_parser.add_argument("--max-capture-queue", type=int, default=4)
    replay_parser.add_argument("--max-map-queue", type=int, default=2)
    replay_parser.add_argument("--drop-policy", choices=("oldest", "newest"), default="oldest")
    replay_parser.add_argument("--voxel-size-m", type=float, default=0.05)
    replay_parser.add_argument("--truncation-voxels", type=float, default=3.0)
    replay_parser.add_argument("--export-point-cloud", action="store_true")
    replay_parser.add_argument(
        "--wall-clock-pacing",
        action="store_true",
        help="Sleep to approximate target FPS; omitted uses deterministic simulated pacing.",
    )
    replay_parser.set_defaults(handler=_run_live_replay_recording)
    capture_adapters_parser = runtime_subparsers.add_parser(
        "capture-adapters",
        help="Inspect dependency-safe runtime capture adapters.",
    )
    capture_adapters_subparsers = capture_adapters_parser.add_subparsers(
        dest="capture_adapters_command",
        required=True,
    )
    capture_adapters_list = capture_adapters_subparsers.add_parser(
        "list",
        help="List runtime capture adapters and dependency status.",
    )
    capture_adapters_list.set_defaults(handler=_run_capture_adapters_list)
    stress_parser = runtime_subparsers.add_parser(
        "sparse-tsdf-stress",
        help="Estimate apartment-scale dense memory and sparse block TSDF state.",
    )
    stress_parser.add_argument("--output", type=Path, required=True)
    stress_parser.add_argument("--room-size-m", default="10,10,3")
    stress_parser.add_argument("--voxel-size-m", type=float, default=0.05)
    stress_parser.add_argument("--truncation-voxels", type=float, default=3.0)
    stress_parser.add_argument("--observation-count", type=int, default=3)
    stress_parser.set_defaults(handler=_run_sparse_tsdf_stress)


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
                mode=args.mode,
                backend=args.backend,
            )
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_live_replay_recording(args: argparse.Namespace) -> int:
    from atlas3r.runtime.live_replay import LiveReplayConfig, run_live_replay_recording

    try:
        result = run_live_replay_recording(
            LiveReplayConfig(
                recording=args.recording,
                output=args.output,
                target_fps=args.target_fps,
                max_frames=args.max_frames,
                mapper_backend=args.mapper_backend,
                map_keyframe_stride=args.map_keyframe_stride,
                max_capture_queue=args.max_capture_queue,
                max_map_queue=args.max_map_queue,
                drop_policy=args.drop_policy,
                voxel_size_m=args.voxel_size_m,
                truncation_voxels=args.truncation_voxels,
                export_point_cloud=args.export_point_cloud,
                wall_clock_pacing=args.wall_clock_pacing,
            )
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_capture_adapters_list(_args: argparse.Namespace) -> int:
    from atlas3r.runtime.capture_adapters import list_capture_adapters

    print("name\tavailable\tdetails")
    for status in list_capture_adapters():
        detail = status.reason or status.install_hint or ""
        print(f"{status.name}\t{status.available}\t{detail}")
    return 0


def _run_sparse_tsdf_stress(args: argparse.Namespace) -> int:
    from atlas3r.runtime.sparse_tsdf_stress import (
        SparseTSDFStressConfig,
        parse_room_size_m,
        run_sparse_tsdf_stress,
    )

    try:
        result = run_sparse_tsdf_stress(
            SparseTSDFStressConfig(
                output=args.output,
                room_size_m=parse_room_size_m(args.room_size_m),
                voxel_size_m=args.voxel_size_m,
                truncation_voxels=args.truncation_voxels,
                observation_count=args.observation_count,
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
