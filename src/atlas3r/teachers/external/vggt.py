"""Real VGGT external teacher runner for Atlas3R teacher-signal caches."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import cast

from atlas3r.forge.clip_cache import (
    load_clip_cache_manifest,
    manifest_path_from_input,
    read_clip_payload_from_entry,
    validate_clip_payload,
)
from atlas3r.teachers.external.cache_writer import (
    ExternalSignalPayload,
    write_external_teacher_signal_cache,
)
from atlas3r.teachers.external.contracts import (
    ExternalTeacherDependencyError,
    ExternalTeacherRunConfig,
    ExternalTeacherStatus,
)
from atlas3r.teachers.external.vggt_arrays import (
    ALIGNMENT_POLICIES,
    VGGTRunStats,
    normalise_vggt_clip_prediction,
)
from atlas3r.teachers.external.vggt_model import (
    INSTALL_HINT,
    VGGTRawClipPredictor,
    load_vggt_predictor,
)
from atlas3r.teachers.vggt_eval import VGGTEvaluationConfig, evaluate_vggt_teacher_signals

TEACHER_NAME = "vggt"
TEACHER_VERSION = "external-vggt-v1"
TEACHER_SOURCE_TYPE = "external_multiview_geometry_teacher"
REPO_ENV_VAR = "ATLAS3R_VGGT_REPO"
CHECKPOINT_ENV_VAR = "ATLAS3R_VGGT_CHECKPOINT"
INPUT_FORMAT = "Atlas3R clip cache with RGB clips, K, source T_world_camera, frame IDs, timestamps"
OUTPUT_FORMAT = "Atlas3R teacher-signal cache v1 with VGGT pseudo-labels"


@dataclass(frozen=True)
class VGGTRunConfig(ExternalTeacherRunConfig):
    device: str = "auto"
    vggt_repo: Path | None = None
    checkpoint: str | None = None
    align_to_source_pose: str = "none"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device: must be one of auto, cuda, mps, or cpu")
        if self.align_to_source_pose not in ALIGNMENT_POLICIES:
            raise ValueError(
                "align_to_source_pose: expected diagnostic_sim3, diagnostic_se3, or none"
            )


class VGGTExternalTeacherRunner:
    """Run VGGT clip-by-clip into Atlas3R teacher-signal payloads."""

    def __init__(self, predictor: VGGTRawClipPredictor | None = None) -> None:
        self._predictor = predictor

    def status(self) -> ExternalTeacherStatus:
        return get_vggt_status()

    def run(self, config: ExternalTeacherRunConfig) -> dict[str, object]:
        vggt_config = _coerce_config(config)
        predictor = self._predictor
        loaded_metadata: dict[str, object]
        if predictor is None:
            status = get_vggt_status(vggt_repo=vggt_config.vggt_repo)
            if status.availability != "available":
                raise ExternalTeacherDependencyError(
                    status.display_name,
                    status.reason or "VGGT is not importable or configured",
                    status.install_hint,
                )
            loaded = load_vggt_predictor(
                vggt_repo=_vggt_repo_from_config(vggt_config),
                checkpoint=_checkpoint_from_config(vggt_config),
                device=vggt_config.device,
            )
            predictor = loaded.predictor
            loaded_metadata = {
                "resolved_device": loaded.resolved_device,
                "model_source": loaded.model_source,
                "checkpoint": loaded.checkpoint,
                "input_resolution": loaded.input_resolution,
            }
        else:
            loaded_metadata = {
                "resolved_device": vggt_config.device,
                "model_source": "injected_test_predictor",
                "checkpoint": vggt_config.checkpoint,
                "input_resolution": None,
            }
        return run_vggt_teacher_signal_cache(
            vggt_config,
            predictor=predictor,
            loaded_metadata=loaded_metadata,
        )


def get_vggt_status(
    *,
    vggt_repo: Path | None = None,
    checkpoint: str | None = None,
) -> ExternalTeacherStatus:
    repo = vggt_repo or _env_path(REPO_ENV_VAR)
    if repo is not None:
        if not repo.is_dir():
            return _unavailable(f"{repo}: configured VGGT repo path is not a directory")
        if not (repo / "vggt").is_dir():
            return _unavailable(f"{repo}: expected a vggt package directory")
        reason = f"VGGT local repo configured by {REPO_ENV_VAR} or --vggt-repo"
        return _available(can_run_locally=True, reason=reason)
    try:
        spec = find_spec("vggt")
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    if spec is None:
        return _unavailable(
            f"missing optional dependency: vggt; set {REPO_ENV_VAR} or install VGGT"
        )
    configured_checkpoint = checkpoint or os.environ.get(CHECKPOINT_ENV_VAR)
    reason = (
        f"VGGT import target is present; checkpoint configured by {CHECKPOINT_ENV_VAR} or CLI"
        if configured_checkpoint
        else "VGGT import target is present; runner will use VGGT.from_pretrained when available"
    )
    return _available(can_run_locally=True, reason=reason)


def run_vggt_teacher_signal_cache(
    config: VGGTRunConfig,
    *,
    predictor: VGGTRawClipPredictor,
    loaded_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Run VGGT, write a validated signal cache, and optionally evaluate it."""

    clip_manifest_path = manifest_path_from_input(config.clip_cache)
    clip_manifest = load_clip_cache_manifest(clip_manifest_path)
    clips = _clip_entries(clip_manifest)
    selected = clips if config.max_clips is None else clips[: config.max_clips]
    if not selected:
        raise ValueError(f"{clip_manifest_path}: no clips selected for VGGT")
    stats = VGGTRunStats()
    payloads: list[ExternalSignalPayload] = []
    for clip_entry in selected:
        clip_payload = read_clip_payload_from_entry(clip_manifest_path.parent, clip_entry)
        validate_clip_payload(
            clip_payload,
            clip_length=_int_field(clip_manifest, "clip_length"),
            height=_int_field(clip_manifest, "image_height"),
            width=_int_field(clip_manifest, "image_width"),
        )
        raw_prediction = predictor(clip_payload)
        payloads.append(
            ExternalSignalPayload(
                source_clip_id=_int_field(clip_entry, "clip_id"),
                arrays=normalise_vggt_clip_prediction(
                    raw_prediction,
                    clip_payload=clip_payload,
                    align_to_source_pose=config.align_to_source_pose,
                    stats=stats,
                ),
            )
        )

    result = write_external_teacher_signal_cache(
        clip_cache=clip_manifest_path,
        output=config.output,
        payloads=payloads,
        teacher_name=TEACHER_NAME,
        teacher_version=TEACHER_VERSION,
        teacher_source_type=TEACHER_SOURCE_TYPE,
        source_metadata={
            "runner": "atlas3r.teachers.external.vggt",
            "model_source": str(loaded_metadata.get("model_source")),
            "checkpoint": _metadata_checkpoint(loaded_metadata.get("checkpoint")),
            "device": str(loaded_metadata.get("resolved_device")),
            "input_resolution": loaded_metadata.get("input_resolution"),
            "alignment_policy": config.align_to_source_pose,
            "pose_source": "VGGT camera pose decoded from external model output",
            "pose_source_type": "external_vggt_pose_with_optional_diagnostic_alignment",
            "measured_geometry": False,
            "pseudo_label": True,
            "known_limitations": [
                "VGGT pose and point maps are pseudo-labels, not measured TUM geometry.",
                "diagnostic_* alignment modes use source clip poses only for evaluation.",
                "Aligned pseudo-labels must not be described as measured geometry.",
            ],
            **stats.source_metadata(),
        },
    )
    result.update(stats.result_fields())
    if config.run_inspect:
        inspect_output = config.inspect_output or config.output.with_name(
            f"{config.output.name}_vggt_eval"
        )
        inspection = evaluate_vggt_teacher_signals(
            VGGTEvaluationConfig(
                clip_cache=clip_manifest_path,
                teacher_cache=config.output,
                output=inspect_output,
                max_clips=config.max_clips or 64,
            )
        )
        result["inspection"] = {
            "summary_path": inspection["summary_path"],
            "per_clip_metrics_path": inspection["per_clip_metrics_path"],
            "markdown_report_path": inspection["markdown_report_path"],
        }
    return result


def _coerce_config(config: ExternalTeacherRunConfig) -> VGGTRunConfig:
    if isinstance(config, VGGTRunConfig):
        return config
    return VGGTRunConfig(
        clip_cache=config.clip_cache,
        output=config.output,
        max_clips=config.max_clips,
        run_inspect=config.run_inspect,
        inspect_output=config.inspect_output,
    )


def _vggt_repo_from_config(config: VGGTRunConfig) -> Path | None:
    return config.vggt_repo or _env_path(REPO_ENV_VAR)


def _checkpoint_from_config(config: VGGTRunConfig) -> str | None:
    if config.checkpoint:
        return config.checkpoint
    env_value = os.environ.get(CHECKPOINT_ENV_VAR)
    return env_value if env_value else None


def _metadata_checkpoint(value: object) -> str | None:
    return None if value is None else str(value)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _available(*, can_run_locally: bool, reason: str) -> ExternalTeacherStatus:
    return ExternalTeacherStatus(
        name=TEACHER_NAME,
        display_name="VGGT",
        availability="available",
        can_run_locally=can_run_locally,
        install_hint=INSTALL_HINT,
        expected_input_format=INPUT_FORMAT,
        expected_output_format=OUTPUT_FORMAT,
        reason=reason,
    )


def _unavailable(reason: str) -> ExternalTeacherStatus:
    return ExternalTeacherStatus(
        name=TEACHER_NAME,
        display_name="VGGT",
        availability="unavailable",
        can_run_locally=False,
        install_hint=INSTALL_HINT,
        expected_input_format=INPUT_FORMAT,
        expected_output_format=OUTPUT_FORMAT,
        reason=reason,
    )


def _clip_entries(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    value = manifest.get("clips")
    if not isinstance(value, list):
        raise ValueError("clip manifest clips: must be a list")
    entries: list[dict[str, object]] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValueError(f"clip manifest.clips[{index}]: must be a mapping")
        entries.append(cast(dict[str, object], entry))
    return entries


def _int_field(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key}: must be an integer")
    return value


__all__ = [
    "CHECKPOINT_ENV_VAR",
    "REPO_ENV_VAR",
    "VGGTExternalTeacherRunner",
    "VGGTRunConfig",
    "get_vggt_status",
    "run_vggt_teacher_signal_cache",
]
