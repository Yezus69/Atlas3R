"""Forge CLI registrations for Atlas3R."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def register_forge_parser(subparsers: Any) -> None:
    forge_parser = subparsers.add_parser(
        "forge",
        help="Create reusable Atlas3R training-data caches.",
    )
    forge_subparsers = forge_parser.add_subparsers(dest="forge_command", required=True)
    tum_parser = forge_subparsers.add_parser(
        "tum-rgbd-clips",
        help="Forge a real TUM RGB-D multi-view clip cache.",
    )
    tum_parser.add_argument("--manifest", type=Path, required=True)
    tum_parser.add_argument("--output", type=Path, required=True)
    tum_parser.add_argument("--split", choices=("train", "val"), required=True)
    tum_parser.add_argument("--clip-length", type=int, default=5)
    tum_parser.add_argument("--stride", type=int, default=1)
    tum_parser.add_argument("--width", type=int, default=160)
    tum_parser.add_argument("--height", type=int, default=120)
    tum_parser.add_argument("--max-clips", type=int, default=None)
    tum_parser.add_argument("--max-frame-gap-s", type=float, default=0.12)
    tum_parser.add_argument("--write-pointmaps", action="store_true")
    tum_parser.add_argument("--write-normals", action="store_true")
    tum_parser.set_defaults(handler=_run_forge_tum_rgbd_clips)


def _run_forge_tum_rgbd_clips(args: argparse.Namespace) -> int:
    from atlas3r.data.image_runtime import PillowDependencyError
    from atlas3r.forge.tum_rgbd_clips import (
        TumRgbdClipForgeConfig,
        forge_tum_rgbd_clip_cache,
    )

    try:
        result = forge_tum_rgbd_clip_cache(
            TumRgbdClipForgeConfig(
                manifest=args.manifest,
                output=args.output,
                split=args.split,
                clip_length=args.clip_length,
                stride=args.stride,
                width=args.width,
                height=args.height,
                max_clips=args.max_clips,
                max_frame_gap_s=args.max_frame_gap_s,
                write_pointmaps=args.write_pointmaps,
                write_normals=args.write_normals,
            )
        )
    except (PillowDependencyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "register_forge_parser",
]
