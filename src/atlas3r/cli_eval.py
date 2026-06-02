"""Evaluation CLI registrations for Atlas3R."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def register_eval_parser(subparsers: Any) -> None:
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run diagnostic evaluation commands.",
    )
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)
    tum_parser = eval_subparsers.add_parser(
        "tum-rgbd-checkpoint",
        help="Evaluate a tiny checkpoint on held-out TUM RGB-D frames.",
    )
    tum_parser.add_argument("--checkpoint", type=Path, required=True)
    tum_parser.add_argument("--manifest", type=Path, required=True)
    tum_parser.add_argument("--output", type=Path, required=True)
    tum_parser.add_argument("--split", default="val")
    tum_parser.add_argument("--width", type=int, default=160)
    tum_parser.add_argument("--height", type=int, default=120)
    tum_parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="cuda",
    )
    tum_parser.add_argument("--max-frames", type=int, default=64)
    tum_parser.add_argument("--num-workers", type=int, default=2)
    tum_parser.add_argument("--write-tsdf", action="store_true")
    tum_parser.add_argument("--voxel-size-m", type=float, default=0.1)
    tum_parser.add_argument("--truncation-voxels", type=float, default=3.0)
    tum_parser.add_argument("--map-max-points", type=int, default=3000)
    tum_parser.set_defaults(handler=_run_eval_tum_rgbd_checkpoint)


def _run_eval_tum_rgbd_checkpoint(args: argparse.Namespace) -> int:
    from atlas3r.data.image_runtime import PillowDependencyError
    from atlas3r.eval.tum_rgbd_checkpoint import (
        TumRgbdCheckpointEvalConfig,
        run_tum_rgbd_checkpoint_eval,
    )
    from atlas3r.training.torch_runtime import TorchDependencyError

    try:
        result = run_tum_rgbd_checkpoint_eval(
            TumRgbdCheckpointEvalConfig(
                checkpoint=args.checkpoint,
                manifest=args.manifest,
                output=args.output,
                split=args.split,
                width=args.width,
                height=args.height,
                device=args.device,
                max_frames=args.max_frames,
                num_workers=args.num_workers,
                write_tsdf=args.write_tsdf,
                voxel_size_m=args.voxel_size_m,
                truncation_voxels=args.truncation_voxels,
                map_max_points=args.map_max_points,
            )
        )
    except (PillowDependencyError, TorchDependencyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "register_eval_parser",
]
