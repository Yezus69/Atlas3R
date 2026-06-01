"""Command-line interface for Atlas3R."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from atlas3r import __version__


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
    subparsers.add_parser("smoke", help="Show the skeleton smoke-test command surface.")
    subparsers.add_parser("profile", help="Show the skeleton profiling command surface.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Atlas3R CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is not None:
        parser.error(f"`{args.command}` is not implemented in the skeleton yet")

    return 0
