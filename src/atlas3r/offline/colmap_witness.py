"""Dependency-safe COLMAP/GLOMAP witness orchestration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from atlas3r.offline.classical_geometry import (
    ClassicalWitnessResult,
    classical_truth_boundary,
    write_classical_status,
)
from atlas3r.offline.colmap_import import read_colmap_text_model, write_colmap_model_artifacts
from atlas3r.offline.colmap_process import (
    ColmapWitnessOptions,
    CommandRunner,
    best_sparse_model_dir,
    colmap_command_plan,
    colmap_install_hint,
    resolve_executable,
    run_dense_if_requested,
    run_glomap_if_requested,
    run_manifest_payload,
    with_required_artifacts,
    write_sparse_summary,
)
from atlas3r.offline.frame_cache import FrameCacheResult, FrameRecord
from atlas3r.offline.keyframes import KeyframeRecord
from atlas3r.offline.run_manifest import FailurePoint, write_json, write_jsonl


def run_colmap_witness(
    run_dir: str | Path,
    *,
    frame_cache: FrameCacheResult,
    keyframes: tuple[KeyframeRecord, ...],
    options: ColmapWitnessOptions,
    failure_points: list[FailurePoint],
) -> ClassicalWitnessResult:
    root = Path(run_dir)
    classical_dir = root / "classical"
    classical_dir.mkdir(parents=True, exist_ok=True)
    if options.colmap_proposal_cache or options.glomap_proposal_cache:
        result = _load_replay(root, options)
        return write_classical_status(root, with_required_artifacts(root, result))
    if not options.enable_colmap and not options.enable_glomap:
        result = ClassicalWitnessResult(
            status="disabled",
            source="none",
            reason="COLMAP/GLOMAP witness was not enabled",
            install_hint="Use --enable-colmap, --enable-glomap, or a proposal cache.",
        )
        return write_classical_status(root, with_required_artifacts(root, result))
    if options.colmap_image_stride <= 0:
        raise ValueError("colmap_image_stride must be positive")
    if options.colmap_max_images is not None and options.colmap_max_images <= 0:
        raise ValueError("colmap_max_images must be positive when provided")
    selected = _select_keyframes(keyframes, options)
    if not selected:
        result = _failed(
            "prepare_images",
            (),
            None,
            "no keyframes available for classical reconstruction",
            "no selected keyframes",
            "decode frames and select keyframes before enabling COLMAP",
        )
        _append_failure(failure_points, result)
        return write_classical_status(root, with_required_artifacts(root, result))
    colmap_exe = resolve_executable(options.colmap_exe, "colmap")
    if colmap_exe is None:
        result = ClassicalWitnessResult(
            status="unavailable",
            source="colmap",
            reason=f"COLMAP executable not found: {options.colmap_exe}",
            install_hint=colmap_install_hint(),
            likely_reason="COLMAP not installed or not on PATH",
            next_debug_action="Install COLMAP or pass --colmap-exe with the full executable path.",
        )
        _append_failure(failure_points, result)
        return write_classical_status(root, with_required_artifacts(root, result))
    records_by_frame = {record.frame_id: record for record in frame_cache.records}
    image_dir = classical_dir / "images"
    _copy_images(root, image_dir, selected, records_by_frame)
    colmap_root = classical_dir / "colmap"
    sparse_dir = colmap_root / "sparse"
    sparse_text = colmap_root / "sparse_text"
    dense_dir = colmap_root / "dense"
    database = colmap_root / "database.db"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, object]] = []
    manifest = run_manifest_payload(options, selected, colmap_exe, image_dir, database)
    write_json(classical_dir / "colmap_run_manifest.json", manifest)
    runner = CommandRunner(classical_dir, commands, timeout_s=options.colmap_timeout_s)
    for stage, cmd in colmap_command_plan(
        colmap_exe,
        image_dir=image_dir,
        database_path=database,
        sparse_dir=sparse_dir,
        camera_model=options.colmap_camera_model,
        matcher=options.colmap_matcher,
        use_gpu=options.colmap_use_gpu,
    ):
        failure = runner.run(stage, cmd)
        if failure is not None:
            result = _failed_from_stage(failure)
            _append_failure(failure_points, result)
            return write_classical_status(root, with_required_artifacts(root, result))
    glomap_status = run_glomap_if_requested(
        classical_dir, options, image_dir=image_dir, database=database, commands=commands
    )
    converter = [
        colmap_exe,
        "model_converter",
        "--input_path",
        str(best_sparse_model_dir(sparse_dir)),
        "--output_path",
        str(sparse_text),
        "--output_type",
        "TXT",
    ]
    failure = runner.run("model_converter", converter)
    if failure is not None:
        result = _failed_from_stage(failure)
        _append_failure(failure_points, result)
        return write_classical_status(root, with_required_artifacts(root, result))
    try:
        model = read_colmap_text_model(sparse_text)
    except ValueError as exc:
        result = _failed(
            "sparse_import",
            tuple(converter),
            None,
            str(exc),
            _likely_reason(str(exc)),
            "Inspect classical/colmap/stdout/stderr and try exhaustive matching or fewer images.",
        )
        _append_failure(failure_points, result)
        return write_classical_status(root, with_required_artifacts(root, result))
    artifacts = write_colmap_model_artifacts(model, classical_dir)
    dense_paths = run_dense_if_requested(
        runner,
        colmap_exe,
        image_dir=image_dir,
        sparse_model_dir=best_sparse_model_dir(sparse_dir),
        dense_dir=dense_dir,
        classical_dir=classical_dir,
        options=options,
    )
    summary = write_sparse_summary(classical_dir, model, sparse_text)
    result = ClassicalWitnessResult(
        status="available",
        source="colmap",
        available=True,
        reason="COLMAP produced registered cameras and sparse points",
        install_hint="COLMAP witness ran successfully.",
        registered_image_count=model.registered_image_count,
        sparse_point_count=model.sparse_point_count,
        dense_fused_ply_path=dense_paths.get("dense_fused"),
        dense_mesh_ply_path=dense_paths.get("dense_mesh"),
        model_text_path="classical/colmap/sparse_text",
        cameras_jsonl_path=artifacts["colmap_cameras_jsonl"],
        images_jsonl_path=artifacts["colmap_images_jsonl"],
        points_npz_path=artifacts["colmap_points3d_npz"],
        sparse_points_ply_path=artifacts["colmap_sparse_points_ply"],
        run_manifest_path="classical/colmap_run_manifest.json",
        command_log_path="classical/colmap_commands.jsonl",
        stdout_tail_path="classical/colmap_stdout_tail.txt",
        stderr_tail_path="classical/colmap_stderr_tail.txt",
        sparse_summary_path=summary,
        glomap_status=glomap_status,
        metadata={
            "selected_image_count": len(selected),
            "truth_boundary": classical_truth_boundary("colmap"),
        },
        sparse_model=model,
    )
    write_jsonl(classical_dir / "colmap_commands.jsonl", commands)
    return write_classical_status(root, with_required_artifacts(root, result))


def _load_replay(root: Path, options: ColmapWitnessOptions) -> ClassicalWitnessResult:
    cache_path = Path(options.colmap_proposal_cache or options.glomap_proposal_cache or "")
    source = (
        "glomap"
        if options.glomap_proposal_cache and not options.colmap_proposal_cache
        else "colmap"
    )
    model_root = _resolve_cached_text_model(cache_path)
    model = read_colmap_text_model(model_root)
    artifacts = write_colmap_model_artifacts(model, root / "classical")
    summary = write_sparse_summary(root / "classical", model, model_root)
    return ClassicalWitnessResult(
        status="available",
        source=source,
        available=True,
        reason=f"{source.upper()} sparse model replayed from cache",
        install_hint="Replay mode does not require COLMAP/GLOMAP executables.",
        registered_image_count=model.registered_image_count,
        sparse_point_count=model.sparse_point_count,
        model_text_path=str(model_root),
        cameras_jsonl_path=artifacts["colmap_cameras_jsonl"],
        images_jsonl_path=artifacts["colmap_images_jsonl"],
        points_npz_path=artifacts["colmap_points3d_npz"],
        sparse_points_ply_path=artifacts["colmap_sparse_points_ply"],
        sparse_summary_path=summary,
        metadata={
            "replay_source": str(cache_path),
            "truth_boundary": classical_truth_boundary(source),
        },
        sparse_model=model,
    )


def _failed_from_stage(row: dict[str, object]) -> ClassicalWitnessResult:
    stderr = str(row.get("stderr_tail", ""))
    stage = str(row.get("stage", "unknown"))
    command = row.get("command", [])
    command_items = command if isinstance(command, list) else []
    returncode_value = row.get("returncode")
    returncode = int(returncode_value) if isinstance(returncode_value, int) else None
    return _failed(
        stage,
        tuple(str(item) for item in command_items),
        returncode,
        stderr,
        _likely_reason(stderr),
        _next_debug_action(stage),
    )


def _failed(
    stage: str,
    command: tuple[str, ...],
    returncode: int | None,
    stderr_tail: str,
    likely_reason: str,
    next_debug_action: str,
) -> ClassicalWitnessResult:
    return ClassicalWitnessResult(
        status="failed",
        source="colmap",
        reason=f"COLMAP classical SfM failed at {stage}",
        install_hint=colmap_install_hint(),
        stage_failed=stage,
        command=command,
        returncode=returncode,
        stderr_tail=stderr_tail,
        likely_reason=likely_reason,
        next_debug_action=next_debug_action,
        command_log_path="classical/colmap_commands.jsonl",
        stdout_tail_path="classical/colmap_stdout_tail.txt",
        stderr_tail_path="classical/colmap_stderr_tail.txt",
    )


def _append_failure(failure_points: list[FailurePoint], result: ClassicalWitnessResult) -> None:
    failure_points.append(
        FailurePoint(
            module="classical_geometry_witness",
            code=f"classical_{result.status}",
            severity="warning" if result.status == "unavailable" else "error",
            status=result.status,
            why=result.reason,
            dependency_missing=None if result.status == "failed" else "COLMAP/GLOMAP executable",
            future_module=result.next_debug_action or result.install_hint,
            artifact_path="classical/classical_status.json",
        )
    )


def _select_keyframes(
    keyframes: tuple[KeyframeRecord, ...], options: ColmapWitnessOptions
) -> tuple[KeyframeRecord, ...]:
    selected = keyframes[:: options.colmap_image_stride]
    if options.colmap_max_images is not None:
        selected = selected[: options.colmap_max_images]
    return tuple(selected)


def _copy_images(
    run_dir: Path,
    image_dir: Path,
    selected: tuple[KeyframeRecord, ...],
    records_by_frame: dict[int, FrameRecord],
) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for keyframe in selected:
        record = records_by_frame[keyframe.frame_id]
        source = run_dir / record.frame_path
        target = image_dir / f"frame_{record.frame_id:06d}{source.suffix.lower() or '.ppm'}"
        shutil.copy2(source, target)


def _resolve_cached_text_model(path: Path) -> Path:
    if (path / "cameras.txt").is_file():
        return path
    if (path / "classical_status.json").is_file():
        status = json.loads((path / "classical_status.json").read_text(encoding="utf-8"))
        model_path = status.get("model_text_path")
        if isinstance(model_path, str):
            candidate = (
                path / Path(model_path).name
                if model_path.startswith("classical/")
                else Path(model_path)
            )
            if (candidate / "cameras.txt").is_file():
                return candidate
    for candidate in (
        path / "colmap" / "sparse_text",
        path / "classical" / "colmap" / "sparse_text",
        path / "sparse_text",
    ):
        if (candidate / "cameras.txt").is_file():
            return candidate
    raise ValueError(f"could not find COLMAP text model under {path}")


def _likely_reason(text: str) -> str:
    lower = text.lower()
    if "database" in lower:
        return "COLMAP database or feature matching failure"
    if "cuda" in lower or "gpu" in lower:
        return "GPU/driver failure"
    if "not enough" in lower or "no good initial image pair" in lower:
        return "too little overlap, low texture, motion blur, or repeated surfaces"
    if "image" in lower and "read" in lower:
        return "image decoding or unsupported image format"
    if not text.strip():
        return "COLMAP returned a nonzero status without stderr"
    return "classical SfM failed; inspect stderr for low texture, blur, overlap, or install issues"


def _next_debug_action(stage: str) -> str:
    if stage in {"sequential_matcher", "mapper"}:
        return (
            "Try --colmap-matcher exhaustive with fewer images, for example --colmap-max-images 48."
        )
    if stage == "feature_extractor":
        return "Verify COLMAP can read the copied images and try --colmap-use-gpu 0."
    return "Inspect classical/colmap_stderr_tail.txt and rerun with fewer images or CPU matching."
