"""M1 canonical asset registry, video inspection, and keyframe proposal."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .contracts import (
    ContractValidationError,
    KeyframeProposal,
    TrackType,
    VideoAsset,
    VideoAssetStatus,
    VideoInspectionReport,
)

DEFAULT_MANIFEST_PATH = Path("config/canonical_assets.json")
DEFAULT_OUTPUT_DIR = Path("runs/m1")
CANONICAL_ASSET_IDS = ("reference_metric", "phone_room")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


class DecoderUnavailable(RuntimeError):
    """Raised when an optional decoder boundary is unavailable."""


@dataclass(frozen=True)
class FrameSample:
    frame_id: int
    source_name: str
    timestamp_s: float | None
    width_px: int
    height_px: int
    gray_small: Any
    sharpness: float
    mean_luma: float
    contrast: float
    dark_clip_fraction: float
    bright_clip_fraction: float


def load_canonical_assets(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    repo_root: str | Path | None = None,
) -> tuple[VideoAsset, ...]:
    root = Path.cwd() if repo_root is None else Path(repo_root)
    path = _resolve_path(Path(manifest_path), root)
    if not path.exists():
        raise FileNotFoundError(f"canonical asset manifest is missing: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    tracks = data.get("tracks")
    if not isinstance(tracks, list):
        raise ContractValidationError("canonical asset manifest must contain tracks[]")

    assets = []
    seen_ids = set()
    for entry in tracks:
        if not isinstance(entry, Mapping):
            raise ContractValidationError("manifest tracks must be mappings")
        asset = VideoAsset(
            asset_id=str(entry.get("asset_id", "")),
            track_type=entry.get("track_type", ""),
            source_uri_or_path=str(entry.get("source_uri_or_path", "")),
            expected_modalities=tuple(entry.get("expected_modalities", ())),
            status=entry.get("status", VideoAssetStatus.UNCHECKED.value),
            metadata=dict(entry.get("metadata", {})),
        )
        if asset.asset_id in seen_ids:
            raise ContractValidationError(f"duplicate asset_id in manifest: {asset.asset_id}")
        seen_ids.add(asset.asset_id)
        assets.append(asset)

    missing_ids = [asset_id for asset_id in CANONICAL_ASSET_IDS if asset_id not in seen_ids]
    if missing_ids:
        raise ContractValidationError(
            f"canonical manifest must register {', '.join(missing_ids)}"
        )
    return tuple(assets)


def run_m1(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    repo_root: str | Path | None = None,
    max_samples: int = 64,
    max_keyframes: int = 8,
) -> dict[str, Any]:
    root = Path.cwd() if repo_root is None else Path(repo_root)
    output_path = _resolve_path(Path(output_dir), root)
    output_path.mkdir(parents=True, exist_ok=True)

    assets = load_canonical_assets(manifest_path, repo_root=root)
    registry_assets = []
    inspections = []
    proposals = []

    for asset in assets:
        inspected_asset, inspection, proposal = inspect_asset(
            asset,
            repo_root=root,
            max_samples=max_samples,
            max_keyframes=max_keyframes,
        )
        registry_assets.append(inspected_asset)
        inspections.append(inspection)
        proposals.append(proposal)
        _write_json(output_path / f"{asset.asset_id}_inspection.json", inspection)
        _write_json(output_path / f"{asset.asset_id}_keyframes.json", proposal)

    registry_report = {
        "manifest_path": str(_resolve_path(Path(manifest_path), root)),
        "report_dir": str(output_path),
        "assets": [_asset_registry_row(asset, report, proposal) for asset, report, proposal in zip(registry_assets, inspections, proposals)],
    }
    _write_json(output_path / "asset_registry_report.json", registry_report)

    return {
        "registry_report": output_path / "asset_registry_report.json",
        "inspection_reports": {
            report.asset_id: output_path / f"{report.asset_id}_inspection.json"
            for report in inspections
        },
        "keyframe_reports": {
            proposal.asset_id: output_path / f"{proposal.asset_id}_keyframes.json"
            for proposal in proposals
        },
        "assets": registry_assets,
        "inspections": inspections,
        "keyframe_proposals": proposals,
    }


def inspect_asset(
    asset: VideoAsset,
    *,
    repo_root: str | Path | None = None,
    max_samples: int = 64,
    max_keyframes: int = 8,
) -> tuple[VideoAsset, VideoInspectionReport, KeyframeProposal]:
    root = Path.cwd() if repo_root is None else Path(repo_root)
    resolved = _resolve_asset_source(asset, root)
    if resolved is None:
        return _unusable_asset(
            asset,
            VideoAssetStatus.UNSUPPORTED,
            "remote_or_non_file_asset_not_available_locally",
            {"source_uri_or_path": asset.source_uri_or_path},
        )
    if not resolved.exists():
        return _unusable_asset(
            asset,
            VideoAssetStatus.MISSING_ASSET,
            "missing_asset",
            {"resolved_path": str(resolved)},
        )

    if resolved.is_dir():
        frame_files = _discover_frame_files(resolved, asset.metadata)
        if frame_files:
            return _inspect_frame_sequence(
                asset,
                resolved,
                frame_files,
                max_samples=max_samples,
                max_keyframes=max_keyframes,
            )
        video_files = _discover_video_files(resolved)
        if video_files:
            selected_video = video_files[0]
            return _inspect_video_file(
                asset,
                selected_video,
                max_samples=max_samples,
                max_keyframes=max_keyframes,
                directory_video_count=len(video_files),
            )
        return _unusable_asset(
            asset,
            VideoAssetStatus.UNSUPPORTED,
            "no_supported_rgb_video_or_frame_sequence_found",
            {"resolved_path": str(resolved)},
        )

    suffix = resolved.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return _inspect_frame_sequence(
            asset,
            resolved.parent,
            (resolved,),
            max_samples=max_samples,
            max_keyframes=max_keyframes,
        )
    if suffix in VIDEO_EXTENSIONS:
        return _inspect_video_file(
            asset,
            resolved,
            max_samples=max_samples,
            max_keyframes=max_keyframes,
        )
    return _unusable_asset(
        asset,
        VideoAssetStatus.UNSUPPORTED,
        "unsupported_asset_file_type",
        {"resolved_path": str(resolved), "suffix": suffix or None},
    )


def _inspect_frame_sequence(
    asset: VideoAsset,
    source_dir: Path,
    frame_files: Sequence[Path],
    *,
    max_samples: int,
    max_keyframes: int,
) -> tuple[VideoAsset, VideoInspectionReport, KeyframeProposal]:
    frame_count = len(frame_files)
    fps_hint = _fps_hint(asset.metadata)
    sample_ids = _evenly_spaced_ids(frame_count, max_samples)
    samples = []
    failed = []
    for frame_id in sample_ids:
        frame_path = frame_files[frame_id]
        timestamp_s = (frame_id / fps_hint) if fps_hint else None
        try:
            samples.append(_load_image_sample(frame_id, frame_path.name, frame_path, timestamp_s))
        except (OSError, DecoderUnavailable) as exc:
            failed.append({"frame_id": frame_id, "source_name": frame_path.name, "error": str(exc)})

    if not samples:
        status = VideoAssetStatus.UNSUPPORTED if failed and "Pillow" in failed[0]["error"] else VideoAssetStatus.CORRUPT
        return _unusable_asset(
            asset,
            status,
            "no_decodable_sampled_frames",
            {
                "resolved_path": str(source_dir),
                "sample_decode_failures": failed[:8],
                "frame_count": frame_count,
            },
        )

    extra_risks = []
    if failed:
        extra_risks.append("some_sampled_frames_failed_decode")
    widths = {sample.width_px for sample in samples}
    heights = {sample.height_px for sample in samples}
    hard_rejections = []
    if len(widths) > 1 or len(heights) > 1:
        hard_rejections.append("inconsistent_sampled_frame_dimensions")

    duration_s = ((frame_count - 1) / fps_hint) if fps_hint and frame_count > 1 else None
    report, proposal = _build_reports(
        asset,
        frame_count=frame_count,
        fps=fps_hint,
        duration_s=duration_s,
        codec_or_container_optional="decoded_frame_sequence",
        samples=samples,
        failed_samples=failed,
        hard_rejections=hard_rejections,
        extra_risks=extra_risks,
        max_keyframes=max_keyframes,
        metadata={
            "source_kind": "decoded_frame_sequence",
            "source_dir": str(source_dir),
            "frame_id_origin": "zero_based_sorted_asset_order",
            "frame_glob": asset.metadata.get("frame_glob"),
            "frame_count_source": "file_count",
        },
    )
    return _asset_with_status(asset, VideoAssetStatus.AVAILABLE, {"resolved_path": str(source_dir)}), report, proposal


def _inspect_video_file(
    asset: VideoAsset,
    video_path: Path,
    *,
    max_samples: int,
    max_keyframes: int,
    directory_video_count: int | None = None,
) -> tuple[VideoAsset, VideoInspectionReport, KeyframeProposal]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        return _unusable_asset(
            asset,
            VideoAssetStatus.UNSUPPORTED,
            "video_decoder_unavailable",
            {"resolved_path": str(video_path), "missing_dependency": exc.name},
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        status = VideoAssetStatus.CORRUPT if video_path.stat().st_size == 0 else VideoAssetStatus.UNSUPPORTED
        return _unusable_asset(
            asset,
            status,
            "video_unreadable_or_unsupported_codec",
            {"resolved_path": str(video_path)},
        )

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or None
    if frame_count <= 0:
        frame_count = _count_video_frames_sequentially(capture, cv2)
        capture.release()
        capture = cv2.VideoCapture(str(video_path))

    sample_ids = _evenly_spaced_ids(frame_count, max_samples)
    samples = []
    failed = []
    for frame_id in sample_ids:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            failed.append({"frame_id": frame_id, "source_name": video_path.name, "error": "cv2_read_failed"})
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp_s = (frame_id / fps) if fps else None
        samples.append(
            _sample_from_rgb_array(
                frame_id,
                video_path.name,
                frame_rgb,
                timestamp_s,
                np,
            )
        )
    capture.release()

    if not samples:
        return _unusable_asset(
            asset,
            VideoAssetStatus.CORRUPT,
            "no_decodable_sampled_video_frames",
            {"resolved_path": str(video_path), "frame_count": frame_count, "sample_decode_failures": failed[:8]},
        )

    extra_risks = []
    if failed:
        extra_risks.append("some_sampled_video_frames_failed_decode")
    if directory_video_count and directory_video_count > 1:
        extra_risks.append("multiple_video_files_found_using_first_sorted_file")

    duration_s = (frame_count / fps) if fps and frame_count > 0 else None
    report, proposal = _build_reports(
        asset,
        frame_count=frame_count,
        fps=fps,
        duration_s=duration_s,
        codec_or_container_optional=f"video_container:{video_path.suffix.lower()}",
        samples=samples,
        failed_samples=failed,
        hard_rejections=[],
        extra_risks=extra_risks,
        max_keyframes=max_keyframes,
        metadata={
            "source_kind": "video_file",
            "source_file": str(video_path),
            "frame_id_origin": "zero_based_video_frame_index",
            "frame_count_source": "container_metadata_or_sequential_count",
        },
    )
    return _asset_with_status(asset, VideoAssetStatus.AVAILABLE, {"resolved_path": str(video_path)}), report, proposal


def _build_reports(
    asset: VideoAsset,
    *,
    frame_count: int,
    fps: float | None,
    duration_s: float | None,
    codec_or_container_optional: str,
    samples: Sequence[FrameSample],
    failed_samples: Sequence[Mapping[str, Any]],
    hard_rejections: Sequence[str],
    extra_risks: Sequence[str],
    max_keyframes: int,
    metadata: Mapping[str, Any],
) -> tuple[VideoInspectionReport, KeyframeProposal]:
    samples_sorted = sorted(samples, key=lambda sample: sample.frame_id)
    motion_by_frame = _motion_by_frame(samples_sorted)
    usable_by_frame = _usable_by_frame(samples_sorted)
    usable_ratio = sum(1 for usable in usable_by_frame.values() if usable) / len(samples_sorted)
    risk_flags = list(extra_risks)

    if fps is None:
        risk_flags.append("timestamps_unavailable")
    if _near_static_ratio(motion_by_frame) > 0.65:
        risk_flags.append("high_near_duplicate_or_static_frame_ratio")
    if _p95(tuple(motion_by_frame.values())) < 0.015 and frame_count > 2:
        risk_flags.append("low_motion_proxy_before_geometry")
    if _p95(tuple(motion_by_frame.values())) > 0.35:
        risk_flags.append("large_scene_change_or_exposure_jump_risk")

    final_hard_rejections = list(hard_rejections)
    if frame_count < 2:
        final_hard_rejections.append("insufficient_frame_count")
    if len(samples_sorted) < 2:
        final_hard_rejections.append("insufficient_decodable_sample_count")
    if usable_ratio < 0.20:
        final_hard_rejections.append("insufficient_usable_frames")

    width = samples_sorted[0].width_px
    height = samples_sorted[0].height_px
    confidence = _clamp01(usable_ratio * min(1.0, len(samples_sorted) / min(frame_count, 64)))
    if final_hard_rejections:
        confidence = min(confidence, 0.35)

    report = VideoInspectionReport(
        asset_id=asset.asset_id,
        frame_count=frame_count,
        fps_or_frame_timestamps={
            "fps": fps,
            "frame_timestamps_s": "derived_from_fps" if fps else None,
            "source": "manifest_or_container" if fps else "unavailable",
        },
        width_px=width,
        height_px=height,
        duration_s=duration_s,
        codec_or_container_optional=codec_or_container_optional,
        sampled_frame_ids=tuple(sample.frame_id for sample in samples_sorted),
        blur_summary={"sharpness_laplacian_variance": _summary(tuple(sample.sharpness for sample in samples_sorted))},
        exposure_summary={
            "mean_luma": _summary(tuple(sample.mean_luma for sample in samples_sorted)),
            "contrast_std_luma": _summary(tuple(sample.contrast for sample in samples_sorted)),
            "dark_clip_fraction": _summary(tuple(sample.dark_clip_fraction for sample in samples_sorted)),
            "bright_clip_fraction": _summary(tuple(sample.bright_clip_fraction for sample in samples_sorted)),
        },
        motion_summary={
            "mean_abs_luma_delta": _summary(tuple(motion_by_frame.values())),
            "near_static_sample_pair_ratio": _near_static_ratio(motion_by_frame),
        },
        scene_change_summary={
            "sample_pair_scene_change_score": _summary(tuple(motion_by_frame.values())),
            "score_note": "M1 uses luma frame-difference proxy only; no geometry baseline is inferred.",
        },
        usable_frame_ratio=usable_ratio,
        hard_rejection_reasons=tuple(_dedupe(final_hard_rejections)),
        soft_risk_flags=tuple(_dedupe(risk_flags)),
        confidence=confidence,
        metadata={
            **dict(metadata),
            "sample_count": len(samples_sorted),
            "sample_decode_failures": list(failed_samples[:8]),
            "quality_thresholds": {
                "sharpness_laplacian_variance_min": 0.00003,
                "contrast_std_luma_min": 0.03,
                "max_dark_or_bright_clip_fraction": 0.35,
            },
        },
    )
    proposal = _propose_keyframes(
        report,
        samples_sorted,
        motion_by_frame,
        usable_by_frame,
        max_keyframes=max_keyframes,
    )
    return report, proposal


def _propose_keyframes(
    report: VideoInspectionReport,
    samples: Sequence[FrameSample],
    motion_by_frame: Mapping[int, float],
    usable_by_frame: Mapping[int, bool],
    *,
    max_keyframes: int,
) -> KeyframeProposal:
    base_risks = list(report.soft_risk_flags)
    if report.hard_rejection_reasons:
        return KeyframeProposal(
            asset_id=report.asset_id,
            selected_frame_ids=(),
            timestamps_s=(),
            selection_reasons=(),
            sharpness_scores=(),
            scene_change_scores=(),
            motion_or_baseline_proxy_scores=(),
            coverage_or_overlap_proxy_scores=(),
            risk_flags=tuple(_dedupe((*base_risks, *report.hard_rejection_reasons))),
            metadata={"proposal_status": "not_proposed_due_to_hard_rejection"},
        )

    candidates = [sample for sample in samples if usable_by_frame.get(sample.frame_id, False)]
    if len(candidates) < 2:
        return KeyframeProposal(
            asset_id=report.asset_id,
            selected_frame_ids=(),
            timestamps_s=(),
            selection_reasons=(),
            sharpness_scores=(),
            scene_change_scores=(),
            motion_or_baseline_proxy_scores=(),
            coverage_or_overlap_proxy_scores=(),
            risk_flags=tuple(_dedupe((*base_risks, "insufficient_usable_frames_for_keyframes"))),
            metadata={"proposal_status": "not_proposed_due_to_insufficient_usable_frames"},
        )

    desired = max(2, min(max_keyframes, len(candidates)))
    sharp_norm = _normalized_scores({sample.frame_id: sample.sharpness for sample in candidates})
    motion_norm = _normalized_scores({sample.frame_id: motion_by_frame.get(sample.frame_id, 0.0) for sample in candidates})
    selected: dict[int, tuple[FrameSample, float, str]] = {}
    last_frame_id = max(report.frame_count - 1, 1)

    for segment_index in range(desired):
        start = int(round(segment_index * report.frame_count / desired))
        end = int(round((segment_index + 1) * report.frame_count / desired)) - 1
        center = (start + end) / 2
        segment = [
            sample
            for sample in candidates
            if sample.frame_id not in selected and start <= sample.frame_id <= end
        ]
        if not segment:
            segment = [
                sample
                for sample in candidates
                if sample.frame_id not in selected
            ]
        if not segment:
            break
        best = max(
            segment,
            key=lambda sample: (
                0.65 * sharp_norm[sample.frame_id]
                + 0.25 * motion_norm[sample.frame_id]
                + 0.10 * (1.0 - min(1.0, abs(sample.frame_id - center) / max(1.0, end - start + 1)))
            ),
        )
        coverage = 1.0 - min(1.0, abs(best.frame_id - center) / max(1.0, end - start + 1))
        reason = (
            f"temporal_segment_{segment_index + 1}; "
            f"sharpness={best.sharpness:.6f}; "
            f"motion_proxy={motion_by_frame.get(best.frame_id, 0.0):.6f}; "
            f"source={best.source_name}"
        )
        selected[best.frame_id] = (best, _clamp01(coverage), reason)

    ordered = [selected[frame_id] for frame_id in sorted(selected)]
    selected_sources = {str(sample.frame_id): sample.source_name for sample, _, _ in ordered}
    base_risks.append("m1_keyframes_are_image_evidence_proposals_not_geometry_baselines")

    return KeyframeProposal(
        asset_id=report.asset_id,
        selected_frame_ids=tuple(sample.frame_id for sample, _, _ in ordered),
        timestamps_s=tuple(sample.timestamp_s for sample, _, _ in ordered),
        selection_reasons=tuple(reason for _, _, reason in ordered),
        sharpness_scores=tuple(sharp_norm[sample.frame_id] for sample, _, _ in ordered),
        scene_change_scores=tuple(motion_norm[sample.frame_id] for sample, _, _ in ordered),
        motion_or_baseline_proxy_scores=tuple(motion_norm[sample.frame_id] for sample, _, _ in ordered),
        coverage_or_overlap_proxy_scores=tuple(coverage for _, coverage, _ in ordered),
        risk_flags=tuple(_dedupe(base_risks)),
        metadata={
            "proposal_status": "proposed",
            "selected_frame_sources": selected_sources,
            "selection_basis": "sampled sharpness, exposure usability, luma motion proxy, and temporal coverage",
        },
    )


def _load_image_sample(
    frame_id: int,
    source_name: str,
    frame_path: Path,
    timestamp_s: float | None,
) -> FrameSample:
    np, Image, ImageOps = _image_deps()
    with Image.open(frame_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return _sample_from_pil_image(frame_id, source_name, image, timestamp_s, np)


def _sample_from_pil_image(
    frame_id: int,
    source_name: str,
    image: Any,
    timestamp_s: float | None,
    np: Any,
) -> FrameSample:
    width, height = image.size
    small = image.resize((96, 96))
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    return _sample_from_rgb_float(frame_id, source_name, rgb, timestamp_s, width, height, np)


def _sample_from_rgb_array(
    frame_id: int,
    source_name: str,
    frame_rgb: Any,
    timestamp_s: float | None,
    np: Any,
) -> FrameSample:
    height, width = frame_rgb.shape[:2]
    # OpenCV already decoded RGB; use its area resize for a fixed proxy grid.
    import cv2  # type: ignore

    resized = cv2.resize(frame_rgb, (96, 96), interpolation=cv2.INTER_AREA)
    rgb = resized.astype(np.float32) / 255.0
    return _sample_from_rgb_float(frame_id, source_name, rgb, timestamp_s, width, height, np)


def _sample_from_rgb_float(
    frame_id: int,
    source_name: str,
    rgb: Any,
    timestamp_s: float | None,
    width: int,
    height: int,
    np: Any,
) -> FrameSample:
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    center = gray[1:-1, 1:-1]
    laplacian = (
        4.0 * center
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    return FrameSample(
        frame_id=frame_id,
        source_name=source_name,
        timestamp_s=timestamp_s,
        width_px=int(width),
        height_px=int(height),
        gray_small=gray,
        sharpness=float(np.var(laplacian)),
        mean_luma=float(np.mean(gray)),
        contrast=float(np.std(gray)),
        dark_clip_fraction=float(np.mean(gray <= (2.0 / 255.0))),
        bright_clip_fraction=float(np.mean(gray >= (253.0 / 255.0))),
    )


def _image_deps() -> tuple[Any, Any, Any]:
    try:
        import numpy as np  # type: ignore
        from PIL import Image, ImageOps  # type: ignore
    except ModuleNotFoundError as exc:
        raise DecoderUnavailable(
            f"Pillow and numpy are required for decoded-frame inspection; missing {exc.name}"
        ) from exc
    return np, Image, ImageOps


def _usable_by_frame(samples: Sequence[FrameSample]) -> dict[int, bool]:
    usable = {}
    for sample in samples:
        usable[sample.frame_id] = (
            sample.sharpness >= 0.00003
            and sample.contrast >= 0.03
            and sample.dark_clip_fraction <= 0.35
            and sample.bright_clip_fraction <= 0.35
        )
    return usable


def _motion_by_frame(samples: Sequence[FrameSample]) -> dict[int, float]:
    if not samples:
        return {}
    np = None
    result = {samples[0].frame_id: 0.0}
    previous = samples[0]
    for sample in samples[1:]:
        if np is None:
            import numpy as np  # type: ignore
        result[sample.frame_id] = float(np.mean(np.abs(sample.gray_small - previous.gray_small)))
        previous = sample
    return result


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    clean = sorted(float(value) for value in values)
    if not clean:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    return {
        "count": len(clean),
        "min": clean[0],
        "median": _percentile(clean, 0.50),
        "mean": sum(clean) / len(clean),
        "p95": _percentile(clean, 0.95),
        "max": clean[-1],
    }


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _p95(values: Sequence[float]) -> float:
    return _percentile(sorted(float(value) for value in values), 0.95)


def _near_static_ratio(motion_by_frame: Mapping[int, float]) -> float:
    values = [value for frame_id, value in motion_by_frame.items() if frame_id != min(motion_by_frame or {0: 0})]
    if not values:
        return 0.0
    return sum(1 for value in values if value < 0.01) / len(values)


def _normalized_scores(scores: Mapping[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = sorted(float(value) for value in scores.values())
    low = _percentile(values, 0.05)
    high = _percentile(values, 0.95)
    if high <= low:
        return {frame_id: 0.5 for frame_id in scores}
    return {
        frame_id: _clamp01((float(value) - low) / (high - low))
        for frame_id, value in scores.items()
    }


def _evenly_spaced_ids(frame_count: int, max_samples: int) -> tuple[int, ...]:
    if frame_count <= 0:
        return ()
    sample_count = max(1, min(frame_count, max_samples))
    if sample_count == 1:
        return (0,)
    ids = {
        int(round(index * (frame_count - 1) / (sample_count - 1)))
        for index in range(sample_count)
    }
    return tuple(sorted(ids))


def _discover_frame_files(source_dir: Path, metadata: Mapping[str, Any]) -> tuple[Path, ...]:
    frame_glob = metadata.get("frame_glob")
    if isinstance(frame_glob, str) and frame_glob.strip():
        files = tuple(path for path in source_dir.glob(frame_glob) if path.is_file())
    else:
        files = tuple(
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return tuple(sorted(files, key=_natural_sort_key))


def _discover_video_files(source_dir: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in source_dir.iterdir()
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            ),
            key=_natural_sort_key,
        )
    )


def _natural_sort_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def _fps_hint(metadata: Mapping[str, Any]) -> float | None:
    value = metadata.get("fps")
    if value is None:
        value = metadata.get("fps_hint")
    if value is None:
        return None
    try:
        fps = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError("fps/fps_hint metadata must be numeric") from exc
    if fps <= 0:
        raise ContractValidationError("fps/fps_hint metadata must be positive")
    return fps


def _count_video_frames_sequentially(capture: Any, cv2: Any) -> int:
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            return count
        count += 1


def _unusable_asset(
    asset: VideoAsset,
    status: VideoAssetStatus,
    reason: str,
    metadata: Mapping[str, Any],
) -> tuple[VideoAsset, VideoInspectionReport, KeyframeProposal]:
    updated = _asset_with_status(asset, status, {"availability_reason": reason, **dict(metadata)})
    report = VideoInspectionReport(
        asset_id=asset.asset_id,
        frame_count=0,
        fps_or_frame_timestamps={"fps": None, "frame_timestamps_s": None, "source": "unavailable"},
        width_px=None,
        height_px=None,
        duration_s=None,
        codec_or_container_optional=None,
        sampled_frame_ids=(),
        blur_summary={},
        exposure_summary={},
        motion_summary={},
        scene_change_summary={},
        usable_frame_ratio=0.0,
        hard_rejection_reasons=(reason,),
        soft_risk_flags=(),
        confidence=0.0,
        metadata={"asset_status": status.value, **dict(metadata)},
    )
    proposal = KeyframeProposal(
        asset_id=asset.asset_id,
        selected_frame_ids=(),
        timestamps_s=(),
        selection_reasons=(),
        sharpness_scores=(),
        scene_change_scores=(),
        motion_or_baseline_proxy_scores=(),
        coverage_or_overlap_proxy_scores=(),
        risk_flags=(reason,),
        metadata={"proposal_status": "not_proposed_due_to_unusable_asset", **dict(metadata)},
    )
    return updated, report, proposal


def _asset_with_status(
    asset: VideoAsset,
    status: VideoAssetStatus,
    metadata_updates: Mapping[str, Any],
) -> VideoAsset:
    metadata = {**dict(asset.metadata), **dict(metadata_updates)}
    return VideoAsset(
        asset_id=asset.asset_id,
        track_type=asset.track_type,
        source_uri_or_path=asset.source_uri_or_path,
        expected_modalities=asset.expected_modalities,
        status=status,
        metadata=metadata,
    )


def _asset_registry_row(
    asset: VideoAsset,
    report: VideoInspectionReport,
    proposal: KeyframeProposal,
) -> dict[str, Any]:
    return {
        "asset": _jsonable(asset),
        "usable_for_m1_keyframe_proposal": (
            asset.status is VideoAssetStatus.AVAILABLE
            and not report.hard_rejection_reasons
            and bool(proposal.selected_frame_ids)
        ),
        "hard_rejection_reasons": list(report.hard_rejection_reasons),
        "soft_risk_flags": list(report.soft_risk_flags),
        "selected_keyframe_count": len(proposal.selected_frame_ids),
        "inspection_report": f"{asset.asset_id}_inspection.json",
        "keyframe_report": f"{asset.asset_id}_keyframes.json",
    }


def _resolve_asset_source(asset: VideoAsset, repo_root: Path) -> Path | None:
    parsed = urlparse(asset.source_uri_or_path)
    if parsed.scheme and parsed.scheme not in {"file"}:
        return None
    if parsed.scheme == "file":
        return Path(parsed.path)
    return _resolve_path(Path(asset.source_uri_or_path), repo_root)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    result = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-keyframes", type=int, default=8)
    args = parser.parse_args(argv)

    result = run_m1(
        args.manifest,
        args.output_dir,
        max_samples=args.max_samples,
        max_keyframes=args.max_keyframes,
    )
    summary = {
        "registry_report": result["registry_report"],
        "inspection_reports": result["inspection_reports"],
        "keyframe_reports": result["keyframe_reports"],
    }
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
