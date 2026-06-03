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
        help="Run a Phase 5D temporal checkpoint as a streaming CPU TSDF map diagnostic.",
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
        choices=("oracle", "student-relative", "both"),
        default="oracle",
    )
    stream_parser.add_argument("--voxel-size-m", type=float, default=0.05)
    stream_parser.add_argument("--truncation-voxels", type=float, default=3.0)
    stream_parser.set_defaults(handler=_run_stream_student_map)


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


__all__ = [
    "register_runtime_parser",
]
