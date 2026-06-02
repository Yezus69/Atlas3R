"""Dataset CLI registrations for Atlas3R."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def register_datasets_parser(subparsers: Any) -> None:
    datasets_parser = subparsers.add_parser(
        "datasets",
        help="Prepare small dependency-light dataset manifests.",
    )
    datasets_subparsers = datasets_parser.add_subparsers(dest="datasets_command", required=True)
    tum_parser = datasets_subparsers.add_parser(
        "tum-rgbd",
        help="Download or prepare supported TUM RGB-D sequences.",
    )
    tum_subparsers = tum_parser.add_subparsers(dest="tum_rgbd_command", required=True)
    tum_download_parser = tum_subparsers.add_parser(
        "download",
        help="Download and safely extract a supported TUM RGB-D sequence.",
    )
    tum_download_parser.add_argument("--sequence", default="freiburg1_xyz")
    tum_download_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output folder for the downloaded archive, ground truth, and extracted sequence.",
    )
    tum_download_parser.set_defaults(handler=_run_datasets_tum_rgbd_download)
    tum_prepare_parser = tum_subparsers.add_parser(
        "prepare",
        help="Prepare an Atlas3R TUM RGB-D manifest from an extracted sequence folder.",
    )
    tum_prepare_parser.add_argument("--sequence", default="freiburg1_xyz")
    tum_prepare_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Extracted TUM RGB-D sequence folder containing rgb.txt, depth.txt, "
            "and groundtruth.txt."
        ),
    )
    tum_prepare_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output manifest JSON path.",
    )
    tum_prepare_parser.add_argument("--stride", type=int, default=1)
    tum_prepare_parser.add_argument("--max-frames", type=int, default=None)
    tum_prepare_parser.add_argument("--max-delta-s", type=float, default=0.02)
    tum_prepare_parser.set_defaults(handler=_run_datasets_tum_rgbd_prepare)


def _run_datasets_tum_rgbd_download(args: argparse.Namespace) -> int:
    from atlas3r.data.tum_rgbd import download_tum_rgbd_sequence

    try:
        written_paths = download_tum_rgbd_sequence(args.sequence, args.output)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print("Downloaded TUM RGB-D sequence files:")
    for path in written_paths:
        print(f"  {path}")
    return 0


def _run_datasets_tum_rgbd_prepare(args: argparse.Namespace) -> int:
    from atlas3r.data.tum_rgbd import prepare_tum_rgbd_manifest

    try:
        manifest = prepare_tum_rgbd_manifest(
            args.input,
            args.output,
            sequence=args.sequence,
            stride=args.stride,
            max_frames=args.max_frames,
            max_delta_s=args.max_delta_s,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Wrote TUM RGB-D manifest with {manifest['frame_count']} frames to {args.output}")
    return 0


__all__ = [
    "register_datasets_parser",
]
