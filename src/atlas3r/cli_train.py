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
    tum_train_parser.add_argument(
        "--model",
        choices=("tiny-v1", "tiny-v2"),
        default="tiny-v1",
        help="Tiny model variant; tiny-v1 preserves existing checkpoint behavior.",
    )
    tum_train_parser.add_argument(
        "--depth-loss",
        choices=("metric_l1", "log_l1"),
        default="metric_l1",
        help="Masked depth loss mode for valid TUM RGB-D pixels.",
    )
    tum_train_parser.add_argument("--hidden-channels", type=int, default=32)
    tum_train_parser.add_argument("--min-valid-depth-pixels", type=int, default=1)
    tum_train_parser.set_defaults(handler=_run_train_tum_rgbd_depth_pose)
    temporal_parser = train_subparsers.add_parser(
        "tum-rgbd-temporal",
        help="Train the tiny temporal debug model on forged TUM RGB-D clips.",
    )
    temporal_parser.add_argument(
        "--clip-cache",
        type=Path,
        required=True,
        help="Input clip-cache manifest JSON or cache directory.",
    )
    temporal_parser.add_argument(
        "--val-clip-cache",
        type=Path,
        default=None,
        help="Optional validation clip-cache manifest JSON or cache directory.",
    )
    temporal_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output run folder for metrics, checkpoints, and preview artifacts.",
    )
    temporal_parser.add_argument("--steps", type=int, default=12000)
    temporal_parser.add_argument("--batch-size", type=int, default=8)
    temporal_parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="cuda",
    )
    temporal_parser.add_argument("--num-workers", type=int, default=4)
    temporal_parser.add_argument("--learning-rate", type=float, default=0.0003)
    temporal_parser.add_argument("--log-every", type=int, default=50)
    temporal_parser.add_argument("--val-every", type=int, default=500)
    temporal_parser.add_argument("--checkpoint-every", type=int, default=1000)
    temporal_parser.add_argument("--preview-every", type=int, default=1000)
    temporal_parser.add_argument("--seed", type=int, default=0)
    temporal_parser.add_argument("--amp", action="store_true")
    temporal_parser.add_argument("--max-runtime-minutes", type=float, default=330.0)
    temporal_parser.add_argument("--hidden-channels", type=int, default=32)
    temporal_parser.add_argument(
        "--depth-loss",
        choices=("metric_l1", "log_l1"),
        default="metric_l1",
    )
    temporal_parser.set_defaults(handler=_run_train_tum_rgbd_temporal)

    teacher_temporal_parser = train_subparsers.add_parser(
        "teacher-signals-temporal",
        help="Train temporal-v1 from measured and pseudo teacher-signal caches.",
    )
    teacher_temporal_parser.add_argument(
        "--teacher-cache",
        type=Path,
        action="append",
        required=True,
        help="Teacher-signal cache manifest or directory. May be supplied more than once.",
    )
    teacher_temporal_parser.add_argument(
        "--val-teacher-cache",
        type=Path,
        action="append",
        default=[],
        help="Optional validation teacher-signal cache. May be supplied more than once.",
    )
    teacher_temporal_parser.add_argument("--output", type=Path, required=True)
    teacher_temporal_parser.add_argument(
        "--model",
        choices=("temporal-v1",),
        default="temporal-v1",
    )
    teacher_temporal_parser.add_argument("--steps", type=int, default=12000)
    teacher_temporal_parser.add_argument("--batch-size", type=int, default=8)
    teacher_temporal_parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="cuda",
    )
    teacher_temporal_parser.add_argument("--num-workers", type=int, default=0)
    teacher_temporal_parser.add_argument("--learning-rate", type=float, default=0.0003)
    teacher_temporal_parser.add_argument("--log-every", type=int, default=50)
    teacher_temporal_parser.add_argument("--val-every", type=int, default=500)
    teacher_temporal_parser.add_argument("--checkpoint-every", type=int, default=1000)
    teacher_temporal_parser.add_argument("--preview-every", type=int, default=1000)
    teacher_temporal_parser.add_argument("--seed", type=int, default=0)
    teacher_temporal_parser.add_argument("--amp", action="store_true")
    teacher_temporal_parser.add_argument("--max-runtime-minutes", type=float, default=330.0)
    teacher_temporal_parser.add_argument("--hidden-channels", type=int, default=24)
    teacher_temporal_parser.add_argument("--bottleneck-channels", type=int, default=32)
    teacher_temporal_parser.add_argument("--min-sigma-m", type=float, default=0.001)
    teacher_temporal_parser.add_argument("--max-sigma-m", type=float, default=1.0)
    teacher_temporal_parser.add_argument("--max-pixel-weight", type=float, default=100.0)
    teacher_temporal_parser.add_argument("--measured-teacher-weight", type=float, default=1.0)
    teacher_temporal_parser.add_argument("--pseudo-teacher-weight", type=float, default=0.25)
    teacher_temporal_parser.add_argument("--relative-translation-weight", type=float, default=0.1)
    teacher_temporal_parser.add_argument("--relative-rotation-weight", type=float, default=0.1)
    teacher_temporal_parser.add_argument("--se3-pose-weight", type=float, default=1.0)
    teacher_temporal_parser.set_defaults(handler=_run_train_teacher_signals_temporal)
    smgt_parser = train_subparsers.add_parser(
        "smgt-tiny",
        help="Train the first diagnostic SMGT-tiny student from a teacher temporal cache.",
    )
    smgt_parser.add_argument("--teacher-cache", type=Path, required=True)
    smgt_parser.add_argument("--output", type=Path, required=True)
    smgt_parser.add_argument("--steps", type=int, default=2000)
    smgt_parser.add_argument("--batch-size", type=int, default=4)
    smgt_parser.add_argument("--device", default="cuda")
    smgt_parser.add_argument("--amp", action="store_true")
    smgt_parser.add_argument("--learning-rate", type=float, default=1e-4)
    smgt_parser.add_argument("--val-split", type=float, default=0.2)
    smgt_parser.add_argument("--heldout-split", type=float, default=0.0)
    smgt_parser.add_argument("--split-manifest", type=Path, default=None)
    smgt_parser.add_argument("--num-workers", type=int, default=2)
    smgt_parser.add_argument("--save-every", type=int, default=500)
    smgt_parser.add_argument("--seed", type=int, default=0)
    smgt_parser.add_argument("--debug-subset-clips", type=int, default=None)
    smgt_parser.add_argument("--debug-allow-pseudo-weight", type=float, default=None)
    smgt_parser.add_argument("--hidden-dim", type=int, default=64)
    smgt_parser.add_argument("--feature-dim", type=int, default=96)
    smgt_parser.add_argument("--memory-dim", type=int, default=128)
    smgt_parser.set_defaults(handler=_run_train_smgt_tiny)


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
                model=args.model,
                depth_loss=args.depth_loss,
                hidden_channels=args.hidden_channels,
                min_valid_depth_pixels=args.min_valid_depth_pixels,
            )
        )
    except (PillowDependencyError, TorchDependencyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_train_tum_rgbd_temporal(args: argparse.Namespace) -> int:
    from atlas3r.training.torch_runtime import TorchDependencyError
    from atlas3r.training.tum_rgbd_temporal_train import (
        TumRgbdTemporalTrainConfig,
        run_tum_rgbd_temporal_training,
    )

    try:
        result = run_tum_rgbd_temporal_training(
            TumRgbdTemporalTrainConfig(
                clip_cache=args.clip_cache,
                val_clip_cache=args.val_clip_cache,
                output=args.output,
                steps=args.steps,
                batch_size=args.batch_size,
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
                depth_loss=args.depth_loss,
            )
        )
    except (TorchDependencyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_train_teacher_signals_temporal(args: argparse.Namespace) -> int:
    from atlas3r.training.teacher_signal_losses import TeacherSignalLossConfig
    from atlas3r.training.teacher_signal_temporal_train import (
        TeacherSignalTemporalTrainConfig,
        run_teacher_signal_temporal_training,
    )
    from atlas3r.training.torch_runtime import TorchDependencyError

    try:
        result = run_teacher_signal_temporal_training(
            TeacherSignalTemporalTrainConfig(
                teacher_caches=tuple(args.teacher_cache),
                val_teacher_caches=tuple(args.val_teacher_cache),
                output=args.output,
                model=args.model,
                steps=args.steps,
                batch_size=args.batch_size,
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
                bottleneck_channels=args.bottleneck_channels,
                loss_config=TeacherSignalLossConfig(
                    min_sigma_m=args.min_sigma_m,
                    max_sigma_m=args.max_sigma_m,
                    max_pixel_weight=args.max_pixel_weight,
                    measured_teacher_weight=args.measured_teacher_weight,
                    pseudo_teacher_weight=args.pseudo_teacher_weight,
                    relative_translation_weight=args.relative_translation_weight,
                    relative_rotation_weight=args.relative_rotation_weight,
                    se3_pose_weight=args.se3_pose_weight,
                ),
            )
        )
    except (TorchDependencyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_train_smgt_tiny(args: argparse.Namespace) -> int:
    from atlas3r.training.smgt_tiny_train import SMGTTinyTrainConfig, run_smgt_tiny_training
    from atlas3r.training.torch_runtime import TorchDependencyError

    try:
        result = run_smgt_tiny_training(
            SMGTTinyTrainConfig(
                teacher_cache=args.teacher_cache,
                output=args.output,
                steps=args.steps,
                batch_size=args.batch_size,
                device=args.device,
                amp=args.amp,
                learning_rate=args.learning_rate,
                val_split=args.val_split,
                heldout_split=args.heldout_split,
                split_manifest=args.split_manifest,
                num_workers=args.num_workers,
                save_every=args.save_every,
                seed=args.seed,
                debug_subset_clips=args.debug_subset_clips,
                debug_allow_pseudo_weight=args.debug_allow_pseudo_weight,
                hidden_dim=args.hidden_dim,
                feature_dim=args.feature_dim,
                memory_dim=args.memory_dim,
            )
        )
    except (TorchDependencyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "register_train_parser",
]
