"""Command-line interface for Atlas3R."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from atlas3r import __version__
from atlas3r.data.synthetic_cube_room import write_synthetic_cube_room_session
from atlas3r.io.session import validate_session
from atlas3r.visualization.session_preview import write_session_preview


def _run_synthetic_cube_room(args: argparse.Namespace) -> int:
    output = write_synthetic_cube_room_session(args.output)
    print(f"Wrote synthetic cube-room session to {output}")
    return 0


def _run_inspect_session(args: argparse.Namespace) -> int:
    validate_session(args.input)
    written_paths = write_session_preview(args.input, args.output)
    print("Wrote session preview:")
    for path in written_paths:
        print(f"  {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level Atlas3R argument parser."""
    parser = argparse.ArgumentParser(
        prog="atlas3r",
        description="Atlas3R real-time RGB neural metric mapping toolkit.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")
    smoke_parser = subparsers.add_parser("smoke", help="Run smoke-test commands.")
    smoke_subparsers = smoke_parser.add_subparsers(dest="smoke_command", required=True)
    synthetic_parser = smoke_subparsers.add_parser(
        "synthetic-cube-room",
        help="Write a deterministic synthetic cube-room .atlas3r session.",
    )
    synthetic_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output session folder to create or update.",
    )
    synthetic_parser.set_defaults(handler=_run_synthetic_cube_room)
    inspect_parser = subparsers.add_parser("inspect", help="Inspect Atlas3R outputs.")
    inspect_subparsers = inspect_parser.add_subparsers(dest="inspect_command", required=True)
    inspect_session_parser = inspect_subparsers.add_parser(
        "session",
        help="Validate a `.atlas3r` session and write dependency-free previews.",
    )
    inspect_session_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input `.atlas3r` session folder.",
    )
    inspect_session_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output preview folder.",
    )
    inspect_session_parser.set_defaults(handler=_run_inspect_session)
    subparsers.add_parser("profile", help="Show the skeleton profiling command surface.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Atlas3R CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "handler", None)
    if handler is None:
        return 0
    return cast(Callable[[argparse.Namespace], int], handler)(args)
