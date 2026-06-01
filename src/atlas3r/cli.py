"""Command-line interface for Atlas3R."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from atlas3r import __version__
from atlas3r.data.synthetic_cube_room import write_synthetic_cube_room_session
from atlas3r.io.session import validate_session
from atlas3r.mapping.cpu_tsdf import write_tsdf_cube_room_smoke
from atlas3r.models.adapters import list_adapters
from atlas3r.visualization.session_preview import write_session_preview


def _run_synthetic_cube_room(args: argparse.Namespace) -> int:
    output = write_synthetic_cube_room_session(args.output)
    print(f"Wrote synthetic cube-room session to {output}")
    return 0


def _run_tsdf_cube_room(args: argparse.Namespace) -> int:
    written_paths = write_tsdf_cube_room_smoke(args.output)
    print("Wrote TSDF cube-room smoke outputs:")
    for path in written_paths:
        print(f"  {path}")
    return 0


def _run_inspect_session(args: argparse.Namespace) -> int:
    validate_session(args.input)
    written_paths = write_session_preview(args.input, args.output)
    print("Wrote session preview:")
    for path in written_paths:
        print(f"  {path}")
    return 0


def _run_adapters_list(_args: argparse.Namespace) -> int:
    print("name\tstatus\tdetails")
    for status in list_adapters():
        detail = status.reason or status.install_hint or ""
        print(f"{status.name}\t{status.availability}\t{detail}")
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
    tsdf_parser = smoke_subparsers.add_parser(
        "tsdf-cube-room",
        help="Run the deterministic CPU TSDF reference smoke on the synthetic cube-room.",
    )
    tsdf_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output folder for TSDF smoke artifacts.",
    )
    tsdf_parser.set_defaults(handler=_run_tsdf_cube_room)
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
    adapters_parser = subparsers.add_parser(
        "adapters",
        help="Inspect dependency-safe teacher adapter stubs.",
    )
    adapters_subparsers = adapters_parser.add_subparsers(dest="adapters_command", required=True)
    adapters_list_parser = adapters_subparsers.add_parser(
        "list",
        help="List known teacher adapters and availability.",
    )
    adapters_list_parser.set_defaults(handler=_run_adapters_list)
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
