"""Command-line interface for Atlas3R."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from atlas3r import __version__
from atlas3r.data.synthetic_cube_room import write_synthetic_cube_room_session
from atlas3r.io.session import validate_session
from atlas3r.io.teacher_cache_inspection import format_teacher_cache_inspection
from atlas3r.mapping.cpu_tsdf import write_tsdf_cube_room_smoke
from atlas3r.mapping.teacher_cache_replay import write_teacher_cache_tsdf_replay
from atlas3r.mapping.tsdf_output_inspection import (
    TSDF_OUTPUT_INSPECTION_MODES,
    format_tsdf_output_folder_inspection,
)
from atlas3r.mapping.world_map_sidecar import format_world_map_sidecar_inspection
from atlas3r.models.adapters import list_adapters
from atlas3r.models.adapters.runner import AdapterRunError, run_adapter_to_cache
from atlas3r.runtime.fixture_inspection import format_runtime_fixture_inspection
from atlas3r.runtime.scheduler import write_runtime_fixture_smoke
from atlas3r.visualization.session_preview import write_session_preview


def _run_synthetic_cube_room(args: argparse.Namespace) -> int:
    output = write_synthetic_cube_room_session(args.output)
    print(f"Wrote synthetic cube-room session to {output}")
    return 0


def _run_tsdf_cube_room(args: argparse.Namespace) -> int:
    written_paths = write_tsdf_cube_room_smoke(
        args.output,
        write_mesh_sidecar=args.write_mesh_sidecar,
        write_world_map_sidecar=args.write_world_map_sidecar,
    )
    print("Wrote TSDF cube-room smoke outputs:")
    for path in written_paths:
        print(f"  {path}")
    return 0


def _run_teacher_cache_tsdf(args: argparse.Namespace) -> int:
    try:
        written_paths = write_teacher_cache_tsdf_replay(
            args.input,
            args.output,
            write_mesh_sidecar=args.write_mesh_sidecar,
            write_world_map_sidecar=args.write_world_map_sidecar,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print("Wrote teacher-cache TSDF replay outputs:")
    for path in written_paths:
        print(f"  {path}")
    return 0


def _run_runtime_fixture(args: argparse.Namespace) -> int:
    try:
        result = write_runtime_fixture_smoke(args.output)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print("Wrote runtime fixture smoke outputs:")
    for path in [
        result.event_log_path,
        result.summary_path,
        result.session_path,
        result.teacher_cache_path,
        result.tsdf_output_path,
    ]:
        print(f"  {path}")
    return 0


def _run_inspect_session(args: argparse.Namespace) -> int:
    validate_session(args.input)
    written_paths = write_session_preview(args.input, args.output)
    print("Wrote session preview:")
    for path in written_paths:
        print(f"  {path}")
    return 0


def _run_inspect_teacher_cache(args: argparse.Namespace) -> int:
    try:
        print(format_teacher_cache_inspection(args.input), end="")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _run_inspect_tsdf_output(args: argparse.Namespace) -> int:
    try:
        print(format_tsdf_output_folder_inspection(args.input, mode=args.mode), end="")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _run_inspect_runtime_fixture(args: argparse.Namespace) -> int:
    try:
        print(format_runtime_fixture_inspection(args.input), end="")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _run_inspect_world_map(args: argparse.Namespace) -> int:
    try:
        print(format_world_map_sidecar_inspection(args.input), end="")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _run_adapters_list(_args: argparse.Namespace) -> int:
    print("name\tstatus\tdetails")
    for status in list_adapters():
        detail = status.reason or status.install_hint or ""
        print(f"{status.name}\t{status.availability}\t{detail}")
    return 0


def _run_adapters_run(args: argparse.Namespace) -> int:
    try:
        result = run_adapter_to_cache(
            adapter_name=args.adapter,
            input_session=args.input,
            output_cache=args.output,
            store_arrays=args.store_arrays,
        )
    except AdapterRunError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Wrote teacher prediction cache to {result.output_cache}")
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
    tsdf_parser.add_argument(
        "--write-mesh-sidecar",
        action="store_true",
        help="Also write mesh_chunk_sidecar.json from observed TSDF surface samples.",
    )
    tsdf_parser.add_argument(
        "--write-world-map-sidecar",
        action="store_true",
        help=(
            "Also write world_map_sidecar.json by validating and wrapping the observed "
            "MeshChunk sidecar."
        ),
    )
    tsdf_parser.set_defaults(handler=_run_tsdf_cube_room)
    teacher_cache_tsdf_parser = smoke_subparsers.add_parser(
        "teacher-cache-tsdf",
        help="Replay a full-array teacher cache into the CPU TSDF reference smoke path.",
    )
    teacher_cache_tsdf_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input teacher prediction cache with arrays.stored=true.",
    )
    teacher_cache_tsdf_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output folder for teacher-cache TSDF replay artifacts.",
    )
    teacher_cache_tsdf_parser.add_argument(
        "--write-mesh-sidecar",
        action="store_true",
        help="Also write mesh_chunk_sidecar.json from observed TSDF surface samples.",
    )
    teacher_cache_tsdf_parser.add_argument(
        "--write-world-map-sidecar",
        action="store_true",
        help=(
            "Also write world_map_sidecar.json by validating and wrapping the observed "
            "MeshChunk sidecar."
        ),
    )
    teacher_cache_tsdf_parser.set_defaults(handler=_run_teacher_cache_tsdf)
    runtime_fixture_parser = smoke_subparsers.add_parser(
        "runtime-fixture",
        help="Run the deterministic runtime fixture scheduler smoke.",
    )
    runtime_fixture_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output folder for runtime event logs, fixture cache, and TSDF artifacts.",
    )
    runtime_fixture_parser.set_defaults(handler=_run_runtime_fixture)
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
    inspect_teacher_cache_parser = inspect_subparsers.add_parser(
        "teacher-cache",
        help="Validate a teacher prediction cache and print deterministic metadata.",
    )
    inspect_teacher_cache_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input teacher prediction cache folder.",
    )
    inspect_teacher_cache_parser.set_defaults(handler=_run_inspect_teacher_cache)
    inspect_tsdf_output_parser = inspect_subparsers.add_parser(
        "tsdf-output",
        help="Validate a CPU TSDF output folder and print deterministic metadata.",
    )
    inspect_tsdf_output_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CPU TSDF output folder.",
    )
    inspect_tsdf_output_parser.add_argument(
        "--mode",
        choices=TSDF_OUTPUT_INSPECTION_MODES,
        default="complete",
        help=(
            "Inspection strictness: surface allows missing sidecars, mesh requires "
            "mesh_chunk_sidecar.json, and world-map/complete require both sidecars."
        ),
    )
    inspect_tsdf_output_parser.set_defaults(handler=_run_inspect_tsdf_output)
    inspect_runtime_fixture_parser = inspect_subparsers.add_parser(
        "runtime-fixture",
        help="Validate a runtime fixture output folder and print deterministic metadata.",
    )
    inspect_runtime_fixture_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input runtime fixture output folder.",
    )
    inspect_runtime_fixture_parser.set_defaults(handler=_run_inspect_runtime_fixture)
    inspect_world_map_parser = inspect_subparsers.add_parser(
        "world-map",
        help="Validate a WorldMap sidecar and print deterministic metadata.",
    )
    inspect_world_map_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input world_map_sidecar.json file.",
    )
    inspect_world_map_parser.set_defaults(handler=_run_inspect_world_map)
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
    adapters_run_parser = adapters_subparsers.add_parser(
        "run",
        help="Run a teacher adapter into a dependency-light prediction cache.",
    )
    adapters_run_parser.add_argument(
        "--adapter",
        required=True,
        help="Adapter name from `atlas3r adapters list`.",
    )
    adapters_run_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input `.atlas3r` session folder.",
    )
    adapters_run_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output teacher prediction cache folder.",
    )
    adapters_run_parser.add_argument(
        "--store-arrays",
        action="store_true",
        help="Store full tensor payloads under teacher_cache/arrays/frame_<id>.npz.",
    )
    adapters_run_parser.set_defaults(handler=_run_adapters_run)
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
