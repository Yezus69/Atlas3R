"""Teacher-signal CLI registrations for Atlas3R."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def register_teachers_parser(subparsers: Any) -> None:
    teachers_parser = subparsers.add_parser(
        "teachers",
        help="Forge, ingest, inspect, and map Atlas3R teacher-signal caches.",
    )
    teacher_subparsers = teachers_parser.add_subparsers(
        dest="teachers_command",
        required=True,
    )

    forge_parser = teacher_subparsers.add_parser(
        "forge-measured-tum",
        help="Forge measured visible-depth TUM teacher signals from a clip cache.",
    )
    forge_parser.add_argument("--clip-cache", type=Path, required=True)
    forge_parser.add_argument("--output", type=Path, required=True)
    forge_parser.add_argument("--sigma-m", type=float, default=0.01)
    forge_parser.add_argument("--max-clips", type=int, default=None)
    forge_parser.set_defaults(handler=_run_forge_measured_tum)

    ingest_parser = teacher_subparsers.add_parser(
        "ingest-local",
        help="Validate local teacher-signal NPZ payloads into the stable cache format.",
    )
    ingest_parser.add_argument("--clip-cache", type=Path, required=True)
    ingest_parser.add_argument("--input", type=Path, required=True)
    ingest_parser.add_argument("--output", type=Path, required=True)
    ingest_parser.add_argument("--teacher-name", required=True)
    ingest_parser.add_argument("--teacher-version", required=True)
    ingest_parser.add_argument("--pseudo-label", action="store_true")
    ingest_parser.set_defaults(handler=_run_ingest_local)

    depth_pro_parser = teacher_subparsers.add_parser(
        "run-depth-pro",
        help="Run Depth Pro into a validated Atlas3R teacher-signal cache.",
    )
    depth_pro_parser.add_argument("--clip-cache", type=Path, required=True)
    depth_pro_parser.add_argument("--output", type=Path, required=True)
    depth_pro_parser.add_argument("--max-clips", type=int, default=None)
    depth_pro_parser.add_argument(
        "--checkpoint-uri",
        default=None,
        help=(
            "External Depth Pro checkpoint URI/path. If omitted, "
            "ATLAS3R_DEPTH_PRO_CHECKPOINT must be set."
        ),
    )
    depth_pro_parser.add_argument(
        "--inspect-output",
        type=Path,
        default=None,
        help="Optional output folder for the post-run inspect-signals diagnostic.",
    )
    depth_pro_parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Depth Pro inference device.",
    )
    depth_pro_parser.set_defaults(handler=_run_depth_pro)

    vggt_parser = teacher_subparsers.add_parser(
        "ingest-vggt-local",
        help="Ingest local VGGT-style NPZ outputs into the stable teacher-signal format.",
    )
    vggt_parser.add_argument("--clip-cache", type=Path, required=True)
    vggt_parser.add_argument("--input", type=Path, required=True)
    vggt_parser.add_argument("--output", type=Path, required=True)
    vggt_parser.add_argument("--teacher-version", default="local-output-v1")
    vggt_parser.add_argument("--max-clips", type=int, default=None)
    vggt_parser.set_defaults(handler=_run_ingest_vggt_local)

    inspect_parser = teacher_subparsers.add_parser(
        "inspect-signals",
        help="Compare teacher-signal depth and pose against a source clip cache.",
    )
    inspect_parser.add_argument("--clip-cache", type=Path, required=True)
    inspect_parser.add_argument("--teacher-cache", type=Path, required=True)
    inspect_parser.add_argument("--output", type=Path, required=True)
    inspect_parser.add_argument("--max-clips", type=int, default=64)
    inspect_parser.set_defaults(handler=_run_inspect_signals)

    map_parser = teacher_subparsers.add_parser(
        "map-signals",
        help="Fuse teacher-signal depth and pose into CPU TSDF map diagnostics.",
    )
    map_parser.add_argument("--teacher-cache", type=Path, required=True)
    map_parser.add_argument("--output", type=Path, required=True)
    map_parser.add_argument("--max-clips", type=int, default=16)
    map_parser.add_argument("--voxel-size-m", type=float, default=0.05)
    map_parser.add_argument("--truncation-voxels", type=float, default=3.0)
    map_parser.set_defaults(handler=_run_map_signals)

    student_parser = teacher_subparsers.add_parser(
        "run-student-temporal",
        help="Run a Phase 5D temporal checkpoint into a pseudo-label teacher-signal cache.",
    )
    student_parser.add_argument("--checkpoint", type=Path, required=True)
    student_parser.add_argument("--clip-cache", type=Path, required=True)
    student_parser.add_argument("--output", type=Path, required=True)
    student_parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    student_parser.add_argument("--max-clips", type=int, default=None)
    student_parser.set_defaults(handler=_run_student_temporal)


def _run_forge_measured_tum(args: argparse.Namespace) -> int:
    from atlas3r.teachers.measured_tum import (
        MeasuredTumTeacherForgeConfig,
        forge_measured_tum_teacher_signal_cache,
    )

    try:
        result = forge_measured_tum_teacher_signal_cache(
            MeasuredTumTeacherForgeConfig(
                clip_cache=args.clip_cache,
                output=args.output,
                sigma_m=args.sigma_m,
                max_clips=args.max_clips,
            )
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_ingest_local(args: argparse.Namespace) -> int:
    from atlas3r.teachers.measured_tum import (
        LocalTeacherIngestConfig,
        ingest_local_teacher_signal_cache,
    )

    try:
        result = ingest_local_teacher_signal_cache(
            LocalTeacherIngestConfig(
                clip_cache=args.clip_cache,
                input=args.input,
                output=args.output,
                teacher_name=args.teacher_name,
                teacher_version=args.teacher_version,
                pseudo_label=args.pseudo_label,
            )
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_depth_pro(args: argparse.Namespace) -> int:
    from atlas3r.teachers.external import (
        DepthProExternalTeacherRunner,
        DepthProRunConfig,
        ExternalTeacherError,
    )

    try:
        result = DepthProExternalTeacherRunner().run(
            DepthProRunConfig(
                clip_cache=args.clip_cache,
                output=args.output,
                max_clips=args.max_clips,
                run_inspect=True,
                inspect_output=args.inspect_output,
                checkpoint_uri=args.checkpoint_uri,
                device=args.device,
            )
        )
    except (ExternalTeacherError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_ingest_vggt_local(args: argparse.Namespace) -> int:
    from atlas3r.teachers.external import (
        VGGTLocalIngestConfig,
        ingest_vggt_local_teacher_signal_cache,
    )

    try:
        result = ingest_vggt_local_teacher_signal_cache(
            VGGTLocalIngestConfig(
                clip_cache=args.clip_cache,
                input=args.input,
                output=args.output,
                teacher_version=args.teacher_version,
                max_clips=args.max_clips,
            )
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_inspect_signals(args: argparse.Namespace) -> int:
    from atlas3r.teachers.map_eval import TeacherSignalInspectConfig, inspect_teacher_signals

    try:
        result = inspect_teacher_signals(
            TeacherSignalInspectConfig(
                clip_cache=args.clip_cache,
                teacher_cache=args.teacher_cache,
                output=args.output,
                max_clips=args.max_clips,
            )
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_map_signals(args: argparse.Namespace) -> int:
    from atlas3r.teachers.map_eval import TeacherSignalMapConfig, map_teacher_signals

    try:
        result = map_teacher_signals(
            TeacherSignalMapConfig(
                teacher_cache=args.teacher_cache,
                output=args.output,
                max_clips=args.max_clips,
                voxel_size_m=args.voxel_size_m,
                truncation_voxels=args.truncation_voxels,
            )
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_student_temporal(args: argparse.Namespace) -> int:
    from atlas3r.teachers.student_temporal import (
        StudentTemporalTeacherRunConfig,
        run_student_temporal_teacher_signal_cache,
    )
    from atlas3r.training.torch_runtime import TorchDependencyError

    try:
        result = run_student_temporal_teacher_signal_cache(
            StudentTemporalTeacherRunConfig(
                checkpoint=args.checkpoint,
                clip_cache=args.clip_cache,
                output=args.output,
                device=args.device,
                max_clips=args.max_clips,
            )
        )
    except (TorchDependencyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "register_teachers_parser",
]
