"""Lazy Torch dataset for Atlas3R teacher-signal temporal training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.forge.clip_cache import (
    load_clip_cache_manifest,
    manifest_path_from_input,
    read_clip_payload_from_entry,
    validate_clip_payload,
)
from atlas3r.pose.transforms import compose_transforms, invert_transform
from atlas3r.teachers.signals import (
    load_teacher_signal_manifest,
    read_teacher_signal_payload_from_entry,
    teacher_signal_manifest_path_from_input,
    validate_payload_matches_signal_entry,
    validate_teacher_signal_payload,
)
from atlas3r.training.torch_runtime import require_torch


@dataclass(frozen=True)
class TeacherSignalRecord:
    teacher_cache_index: int
    signal_index: int
    source_clip_id: int
    signal_entry: dict[str, object]


class TeacherSignalTemporalDataset:
    """Load aligned RGB clips and teacher signals from validated cache manifests."""

    def __init__(
        self,
        teacher_caches: Sequence[str | Path],
        *,
        record_indices: Sequence[int] | None = None,
    ) -> None:
        if not teacher_caches:
            raise ValueError("teacher_caches: at least one cache is required")
        require_torch()
        self.teacher_cache_paths = tuple(
            teacher_signal_manifest_path_from_input(path) for path in teacher_caches
        )
        self.teacher_roots = tuple(path.parent for path in self.teacher_cache_paths)
        self.teacher_manifests: tuple[dict[str, object], ...] = tuple(
            load_teacher_signal_manifest(path, validate_payloads=False)
            for path in self.teacher_cache_paths
        )
        self.clip_manifest_paths = tuple(
            _source_clip_manifest_path(manifest, root)
            for manifest, root in zip(self.teacher_manifests, self.teacher_roots, strict=True)
        )
        self.clip_manifests = tuple(
            load_clip_cache_manifest(path) for path in self.clip_manifest_paths
        )
        self._validate_compatible_caches()
        self.clip_roots = tuple(path.parent for path in self.clip_manifest_paths)
        self.clip_entries = tuple(
            tuple(_entries(manifest, "clips")) for manifest in self.clip_manifests
        )
        self.records = self._records()
        if record_indices is not None:
            self.records = tuple(self.records[index] for index in record_indices)
        if not self.records:
            raise ValueError("teacher_caches: no teacher signal records selected")
        self.clip_length = _int_field(self.teacher_manifests[0], "clip_length")
        self.height = _int_field(self.teacher_manifests[0], "image_height")
        self.width = _int_field(self.teacher_manifests[0], "image_width")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.records):
            raise IndexError(index)
        torch = require_torch()
        record = self.records[index]
        manifest = self.teacher_manifests[record.teacher_cache_index]
        clip_manifest = self.clip_manifests[record.teacher_cache_index]
        clip_entry = self.clip_entries[record.teacher_cache_index][record.source_clip_id]
        clip_payload = read_clip_payload_from_entry(
            self.clip_roots[record.teacher_cache_index], clip_entry
        )
        validate_clip_payload(
            clip_payload,
            clip_length=self.clip_length,
            height=self.height,
            width=self.width,
        )
        teacher_payload = read_teacher_signal_payload_from_entry(
            self.teacher_roots[record.teacher_cache_index],
            record.signal_entry,
        )
        validate_teacher_signal_payload(
            teacher_payload,
            clip_length=self.clip_length,
            height=self.height,
            width=self.width,
        )
        validate_payload_matches_signal_entry(
            teacher_payload,
            record.signal_entry,
            index=record.signal_index,
        )

        images = np.asarray(clip_payload["images_rgb_u8"], dtype=np.float32) / np.float32(255.0)
        images_chw = np.transpose(images, (0, 3, 1, 2)).copy()
        depth = np.asarray(teacher_payload["depth_m"], dtype=np.float32)
        sigma = np.asarray(teacher_payload["depth_sigma_m"], dtype=np.float32)
        confidence = np.asarray(teacher_payload["confidence"], dtype=np.float32)
        valid_mask = np.asarray(teacher_payload["valid_mask"], dtype=np.bool_) & (sigma > 0.0)
        intrinsics = np.asarray(teacher_payload["K"], dtype=np.float32)
        transforms = np.asarray(teacher_payload["T_world_camera"], dtype=np.float32)
        frame_ids = np.asarray(teacher_payload["frame_ids"], dtype=np.int64)
        timestamps = np.asarray(teacher_payload["timestamps_s"], dtype=np.float32)
        target: dict[str, Any] = {
            "depth_m": torch.from_numpy(depth[:, np.newaxis, :, :].copy()),
            "depth_sigma_m": torch.from_numpy(sigma[:, np.newaxis, :, :].copy()),
            "confidence": torch.from_numpy(confidence[:, np.newaxis, :, :].copy()),
            "valid_mask": torch.from_numpy(valid_mask[:, np.newaxis, :, :].copy()),
            "teacher_is_measured": torch.tensor(_teacher_is_measured(manifest)),
            "teacher_name": str(manifest["teacher_name"]),
            "intrinsics": torch.from_numpy(intrinsics.copy()),
            "relative_translation_center_from_camera": torch.from_numpy(
                _relative_translation_center_from_camera(transforms)
            ),
            "relative_rotation_6d_center_from_camera": torch.from_numpy(
                _relative_rotation_6d_center_from_camera(transforms)
            ),
        }
        if "pointmap_camera_m" in teacher_payload:
            pointmap = np.asarray(teacher_payload["pointmap_camera_m"], dtype=np.float32)
            target["pointmap_camera_m"] = torch.from_numpy(
                np.transpose(pointmap, (0, 3, 1, 2)).copy()
            )
            target["pointmap_camera_valid"] = torch.tensor(True)
        else:
            target["pointmap_camera_m"] = torch.zeros(
                (self.clip_length, 3, self.height, self.width),
                dtype=torch.float32,
            )
            target["pointmap_camera_valid"] = torch.tensor(False)
        return {
            "images_rgb": torch.from_numpy(images_chw.astype(np.float32, copy=False)),
            "intrinsics": torch.from_numpy(intrinsics.copy()),
            "T_world_camera": torch.from_numpy(transforms.copy()),
            "frame_ids": torch.from_numpy(frame_ids.copy()),
            "timestamps_s": torch.from_numpy(timestamps.copy()),
            "target": target,
            "metadata": {
                "teacher_cache": str(self.teacher_cache_paths[record.teacher_cache_index]),
                "teacher_name": str(manifest["teacher_name"]),
                "teacher_source_type": str(manifest["teacher_source_type"]),
                "source_clip_id": record.source_clip_id,
                "signal_id": record.signal_index,
                "clip_cache": str(self.clip_manifest_paths[record.teacher_cache_index]),
                "source_dataset_name": str(clip_manifest["source_dataset_name"]),
                "source_sequence_name": str(clip_manifest["source_sequence_name"]),
                "split": str(clip_manifest["split"]),
            },
        }

    def cache_summary(self) -> dict[str, object]:
        measured = sum(
            1
            for record in self.records
            if _teacher_is_measured(self.teacher_manifests[record.teacher_cache_index])
        )
        return {
            "teacher_caches": [str(path) for path in self.teacher_cache_paths],
            "clip_caches": [str(path) for path in self.clip_manifest_paths],
            "record_count": len(self.records),
            "measured_record_count": measured,
            "pseudo_record_count": len(self.records) - measured,
            "source_dataset_name": str(self.teacher_manifests[0]["source_dataset_name"]),
            "source_sequence_name": str(self.teacher_manifests[0]["source_sequence_name"]),
            "split": str(self.teacher_manifests[0]["split"]),
            "clip_length": self.clip_length,
            "image_width": self.width,
            "image_height": self.height,
        }

    def _records(self) -> tuple[TeacherSignalRecord, ...]:
        records: list[TeacherSignalRecord] = []
        for cache_index, manifest in enumerate(self.teacher_manifests):
            entries = _entries(manifest, "signals")
            ordered = sorted(
                enumerate(entries),
                key=lambda item: (_int_field(item[1], "source_clip_id"), item[0]),
            )
            for signal_index, entry in ordered:
                records.append(
                    TeacherSignalRecord(
                        teacher_cache_index=cache_index,
                        signal_index=signal_index,
                        source_clip_id=_int_field(entry, "source_clip_id"),
                        signal_entry=entry,
                    )
                )
        return tuple(records)

    def _validate_compatible_caches(self) -> None:
        expected = _compatibility_key(self.teacher_manifests[0])
        for manifest in self.teacher_manifests[1:]:
            if _compatibility_key(manifest) != expected:
                raise ValueError("teacher_caches: all caches must target compatible clip geometry")


def teacher_signal_train_val_indices(
    record_count: int,
    *,
    val_fraction: float = 0.2,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return a deterministic tail-block validation split."""

    if record_count <= 0:
        raise ValueError("record_count: must be positive")
    if record_count == 1:
        return (0,), (0,)
    if val_fraction <= 0.0 or val_fraction >= 1.0:
        raise ValueError("val_fraction: must be in (0, 1)")
    val_count = max(1, int(round(record_count * val_fraction)))
    val_count = min(val_count, record_count - 1)
    split = record_count - val_count
    return tuple(range(split)), tuple(range(split, record_count))


def _source_clip_manifest_path(manifest: dict[str, object], cache_root: Path) -> Path:
    value = manifest.get("source_clip_cache_manifest_path")
    if not isinstance(value, str) or not value:
        raise ValueError("source_clip_cache_manifest_path: must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        path = cache_root / path
        if not path.exists():
            cwd_relative = Path(value)
            if cwd_relative.exists():
                path = cwd_relative
    return manifest_path_from_input(path)


def _compatibility_key(manifest: dict[str, object]) -> tuple[object, ...]:
    return (
        manifest["source_dataset_name"],
        manifest["source_sequence_name"],
        manifest["split"],
        manifest["clip_length"],
        manifest["image_width"],
        manifest["image_height"],
    )


def _relative_translation_center_from_camera(
    T_world_camera: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    center_index = T_world_camera.shape[0] // 2
    T_center_world = invert_transform(T_world_camera[center_index])
    relative = [
        compose_transforms(T_center_world, T_world_camera[index])[:3, 3]
        for index in range(T_world_camera.shape[0])
    ]
    return cast(npt.NDArray[np.float32], np.stack(relative).astype(np.float32, copy=False))


def _relative_rotation_6d_center_from_camera(
    T_world_camera: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    center_index = T_world_camera.shape[0] // 2
    T_center_world = invert_transform(T_world_camera[center_index])
    rotations = [
        compose_transforms(T_center_world, T_world_camera[index])[:3, :3]
        for index in range(T_world_camera.shape[0])
    ]
    encoded = [np.concatenate([rotation[:, 0], rotation[:, 1]], axis=0) for rotation in rotations]
    return cast(npt.NDArray[np.float32], np.stack(encoded).astype(np.float32, copy=False))


def _teacher_is_measured(manifest: dict[str, object]) -> bool:
    truth = manifest.get("truth_boundary")
    if not isinstance(truth, dict):
        raise ValueError("truth_boundary: must be a mapping")
    return bool(truth.get("measured_geometry"))


def _entries(manifest: dict[str, object], key: str) -> list[dict[str, object]]:
    value = manifest.get(key)
    if not isinstance(value, list):
        raise ValueError(f"manifest.{key}: must be a list")
    return [cast(dict[str, object], item) for item in value]


def _int_field(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key}: must be an integer")
    return value


__all__ = [
    "TeacherSignalRecord",
    "TeacherSignalTemporalDataset",
    "teacher_signal_train_val_indices",
]
