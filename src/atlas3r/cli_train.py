"""Training CLI registrations for Atlas3R."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def register_train_parser(subparsers: Any) -> None:
    train_parser = subparsers.add_parser(
        "train",
        help="Run bounded optional training MVP commands.",
    )
    train_subparsers = train_parser.add_subparsers(dest="train_command", required=True)
    synthetic_overfit_parser = train_subparsers.add_parser(
        "synthetic-overfit",
        help="Train the tiny synthetic-only depth/pose MVP.",
    )
    synthetic_overfit_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output run folder for checkpoint, metrics, and preview artifacts.",
    )
    synthetic_overfit_parser.add_argument("--steps", type=int, default=200)
    synthetic_overfit_parser.add_argument("--batch-size", type=int, default=8)
    synthetic_overfit_parser.add_argument("--num-samples", type=int, default=64)
    synthetic_overfit_parser.add_argument("--width", type=int, default=64)
    synthetic_overfit_parser.add_argument("--height", type=int, default=48)
    synthetic_overfit_parser.add_argument("--seed", type=int, default=0)
    synthetic_overfit_parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    synthetic_overfit_parser.add_argument("--learning-rate", type=float, default=0.001)
    synthetic_overfit_parser.add_argument("--log-every", type=int, default=10)
    synthetic_overfit_parser.set_defaults(handler=_run_train_synthetic_overfit)
    tum_train_parser = train_subparsers.add_parser(
        "tum-rgbd-depth-pose",
        help="Train the tiny debug model on a TUM RGB-D manifest.",
    )
    tum_train_parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Input Atlas3R TUM RGB-D manifest JSON.",
    )
    tum_train_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output run folder for metrics, checkpoints, and preview artifacts.",
    )
    tum_train_parser.add_argument("--steps", type=int, default=20000)
    tum_train_parser.add_argument("--batch-size", type=int, default=16)
    tum_train_parser.add_argument("--width", type=int, default=160)
    tum_train_parser.add_argument("--height", type=int, default=120)
    tum_train_parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="cuda",
    )
    tum_train_parser.add_argument("--num-workers", type=int, default=4)
    tum_train_parser.add_argument("--learning-rate", type=float, default=0.0005)
    tum_train_parser.add_argument("--log-every", type=int, default=50)
    tum_train_parser.add_argument("--val-every", type=int, default=500)
    tum_train_parser.add_argument("--checkpoint-every", type=int, default=1000)
    tum_train_parser.add_argument("--preview-every", type=int, default=1000)
    tum_train_parser.add_argument("--seed", type=int, default=0)
    tum_train_parser.add_argument("--amp", action="store_true")
    tum_train_parser.add_argument("--max-runtime-minutes", type=float, default=None)
    tum_train_parser.add_argument("--hidden-channels", type=int, default=32)
    tum_train_parser.add_argument("--min-valid-depth-pixels", type=int, default=1)
    tum_train_parser.set_defaults(handler=_run_train_tum_rgbd_depth_pose)


def _run_train_synthetic_overfit(args: argparse.Namespace) -> int:
    from atlas3r.training import SyntheticOverfitConfig, TorchDependencyError, run_synthetic_overfit

    try:
        result = run_synthetic_overfit(
            SyntheticOverfitConfig(
                output=args.output,
                steps=args.steps,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                width=args.width,
                height=args.height,
                seed=args.seed,
                device=args.device,
                learning_rate=args.learning_rate,
                log_every=args.log_every,
            )
        )
    except (TorchDependencyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_train_tum_rgbd_depth_pose(args: argparse.Namespace) -> int:
    from atlas3r.data.image_runtime import PillowDependencyError
    from atlas3r.training.torch_runtime import TorchDependencyError
    from atlas3r.training.tum_rgbd_train import (
        TumRgbdTrainConfig,
        run_tum_rgbd_depth_pose_training,
    )

    try:
        result = run_tum_rgbd_depth_pose_training(
            TumRgbdTrainConfig(
                manifest=args.manifest,
                output=args.output,
                steps=args.steps,
                batch_size=args.batch_size,
                width=args.width,
                height=args.height,
                device=args.device,
                num_workers=args.num_workers,
                learning_rate=args.learning_rate,
                log_every=args.log_every,
                val_every=args.val_every,
                checkpoint_every=args.checkpoint_every,
                preview_every=args.preview_every,
                seed=args.seed,
                amp=args.amp,
                max_runtime_minutes=args.max_runtime_minutes,
                hidden_channels=args.hidden_channels,
                min_valid_depth_pixels=args.min_valid_depth_pixels,
            )
        )
    except (PillowDependencyError, TorchDependencyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "register_train_parser",
]
