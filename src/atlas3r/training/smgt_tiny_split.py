"""Deterministic train/val/heldout split manifests for SMGT-tiny caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from atlas3r.training.teacher_temporal_cache import MANIFEST_FILENAME

FORMAT_NAME = "atlas3r_smgt_tiny_split_manifest"
FORMAT_VERSION = 1
SPLIT_MANIFEST_FILENAME = "smgt_tiny_split_manifest.json"
SPLIT_REPORT_FILENAME = "split_report.md"
SPLIT_NAMES = ("train", "val", "heldout")


def build_smgt_tiny_split_manifest(
    teacher_cache: str | Path,
    *,
    val_split: float,
    heldout_split: float,
) -> dict[str, object]:
    """Build a deterministic non-overlapping temporal split manifest."""

    _validate_split_fractions(val_split=val_split, heldout_split=heldout_split)
    cache = Path(teacher_cache)
    source = _read_teacher_cache_manifest(cache)
    clips = _clips(source)
    all_frame_ids = sorted({frame_id for clip in clips for frame_id in _frame_ids(clip)})
    if not all_frame_ids:
        raise ValueError("teacher cache has no frame IDs")
    frame_sets = _frame_range_sets(
        all_frame_ids,
        val_split=val_split,
        heldout_split=heldout_split,
    )
    splits = {name: _split_record(name, clips, frame_sets[name]) for name in SPLIT_NAMES}
    omitted = [
        _required_int(clip.get("clip_id"), "clip.clip_id")
        for clip in clips
        if not any(set(_frame_ids(clip)).issubset(frame_sets[name]) for name in SPLIT_NAMES)
    ]
    manifest = {
        "format_name": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "teacher_cache": str(cache),
        "source_teacher_cache_format_name": source["format_name"],
        "source_teacher_cache_format_version": source["format_version"],
        "split_policy": "temporal_non_overlapping_frame_ranges",
        "train_split": float(1.0 - val_split - heldout_split),
        "val_split": float(val_split),
        "heldout_split": float(heldout_split),
        "source_clip_count": len(clips),
        "source_frame_count": len(all_frame_ids),
        "source_first_frame_id": all_frame_ids[0],
        "source_last_frame_id": all_frame_ids[-1],
        "splits": splits,
        "omitted_boundary_clip_indices": omitted,
        "no_frame_id_overlap": True,
        "heldout_non_overlapping": True,
    }
    train_record = splits["train"]
    val_record = splits["val"]
    heldout_record = splits["heldout"]
    if _required_int(train_record.get("clip_count"), "splits.train.clip_count") <= 0:
        raise ValueError("train split has no clips")
    if (
        val_split > 0.0
        and _required_int(val_record.get("clip_count"), "splits.val.clip_count") <= 0
    ):
        raise ValueError("val split has no clips")
    if (
        heldout_split > 0.0
        and _required_int(heldout_record.get("clip_count"), "splits.heldout.clip_count") <= 0
    ):
        raise ValueError("heldout split has no clips")
    validate_smgt_tiny_split_manifest(manifest)
    return manifest


def write_smgt_tiny_split_manifest(
    output_dir: str | Path,
    teacher_cache: str | Path,
    *,
    val_split: float,
    heldout_split: float,
) -> dict[str, object]:
    """Write split JSON and a compact Markdown report."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_smgt_tiny_split_manifest(
        teacher_cache,
        val_split=val_split,
        heldout_split=heldout_split,
    )
    _write_json(output / SPLIT_MANIFEST_FILENAME, manifest)
    (output / SPLIT_REPORT_FILENAME).write_text(
        format_smgt_tiny_split_report(manifest), encoding="utf-8"
    )
    return manifest


def load_smgt_tiny_split_manifest(path: str | Path) -> dict[str, object]:
    """Load and validate a split manifest JSON file."""

    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid SMGT-tiny split manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("split manifest: must be a JSON object")
    validate_smgt_tiny_split_manifest(manifest)
    return cast(dict[str, object], manifest)


def validate_smgt_tiny_split_manifest(manifest: dict[str, object]) -> None:
    """Validate split schema and no-overlap invariants."""

    if manifest.get("format_name") != FORMAT_NAME:
        raise ValueError("split manifest format_name: unsupported")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("split manifest format_version: unsupported")
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("splits: must be a mapping")
    frame_sets: dict[str, set[int]] = {}
    for name in SPLIT_NAMES:
        record = splits.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"splits.{name}: missing")
        clip_indices = _int_list(record.get("clip_indices"), f"splits.{name}.clip_indices")
        frame_ids = _int_list(record.get("frame_ids"), f"splits.{name}.frame_ids")
        if len(set(clip_indices)) != len(clip_indices):
            raise ValueError(f"splits.{name}.clip_indices: duplicates are not allowed")
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError(f"splits.{name}.frame_ids: duplicates are not allowed")
        if int(record.get("clip_count", -1)) != len(clip_indices):
            raise ValueError(f"splits.{name}.clip_count: does not match clip_indices")
        if int(record.get("frame_count", -1)) != len(frame_ids):
            raise ValueError(f"splits.{name}.frame_count: does not match frame_ids")
        frame_sets[name] = set(frame_ids)
    for left_index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[left_index + 1 :]:
            overlap = frame_sets[left_name] & frame_sets[right_name]
            if overlap:
                overlap_preview = sorted(overlap)[:8]
                raise ValueError(
                    f"split frame ID overlap between {left_name} and {right_name}: "
                    f"{overlap_preview}"
                )
    if manifest.get("no_frame_id_overlap") is not True:
        raise ValueError("no_frame_id_overlap: must be true")


def split_indices(manifest: dict[str, object], split_name: str) -> tuple[int, ...]:
    """Return source clip indices for one split."""

    if split_name not in SPLIT_NAMES:
        raise ValueError("split_name: must be train, val, or heldout")
    splits = cast(dict[str, object], manifest["splits"])
    record = cast(dict[str, object], splits[split_name])
    return tuple(_int_list(record["clip_indices"], f"splits.{split_name}.clip_indices"))


def format_smgt_tiny_split_report(manifest: dict[str, object]) -> str:
    """Format a compact split report."""

    splits = cast(dict[str, object], manifest["splits"])
    lines = [
        "# SMGT Tiny Split Report",
        "",
        f"- Teacher cache: `{manifest['teacher_cache']}`",
        f"- Split policy: `{manifest['split_policy']}`",
        "- Source clips/frames: "
        f"`{manifest['source_clip_count']}` / `{manifest['source_frame_count']}`",
        f"- No frame ID overlap: `{manifest['no_frame_id_overlap']}`",
        "",
        "| Split | Clips | Frames | First frame | Last frame |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in SPLIT_NAMES:
        record = cast(dict[str, object], splits[name])
        lines.append(
            f"| {name} | {record['clip_count']} | {record['frame_count']} | "
            f"{record['first_frame_id']} | {record['last_frame_id']} |"
        )
    lines.extend(
        [
            "",
            f"- Omitted boundary clips: `{manifest['omitted_boundary_clip_indices']}`",
            "",
        ]
    )
    return "\n".join(lines)


def format_smgt_tiny_split_inspection(
    teacher_cache: str | Path,
    *,
    val_split: float,
    heldout_split: float,
) -> str:
    """Return deterministic JSON for CLI inspection."""

    manifest = build_smgt_tiny_split_manifest(
        teacher_cache,
        val_split=val_split,
        heldout_split=heldout_split,
    )
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def _frame_range_sets(
    frame_ids: list[int],
    *,
    val_split: float,
    heldout_split: float,
) -> dict[str, set[int]]:
    count = len(frame_ids)
    train_end = int(count * (1.0 - val_split - heldout_split))
    val_end = int(count * (1.0 - heldout_split))
    train_end = min(max(train_end, 1), count)
    val_end = min(max(val_end, train_end), count)
    return {
        "train": set(frame_ids[:train_end]),
        "val": set(frame_ids[train_end:val_end]),
        "heldout": set(frame_ids[val_end:]),
    }


def _split_record(
    name: str,
    clips: list[dict[str, object]],
    allowed_frame_ids: set[int],
) -> dict[str, object]:
    selected = [clip for clip in clips if set(_frame_ids(clip)).issubset(allowed_frame_ids)]
    clip_indices = [_required_int(clip.get("clip_id"), "clip.clip_id") for clip in selected]
    frame_ids = sorted({frame_id for clip in selected for frame_id in _frame_ids(clip)})
    return {
        "name": name,
        "clip_indices": clip_indices,
        "clip_count": len(clip_indices),
        "frame_ids": frame_ids,
        "frame_count": len(frame_ids),
        "first_frame_id": None if not frame_ids else frame_ids[0],
        "last_frame_id": None if not frame_ids else frame_ids[-1],
    }


def _validate_split_fractions(*, val_split: float, heldout_split: float) -> None:
    if val_split < 0.0 or heldout_split < 0.0:
        raise ValueError("val_split and heldout_split must be non-negative")
    if val_split + heldout_split >= 1.0:
        raise ValueError("val_split + heldout_split must be < 1")


def _read_teacher_cache_manifest(cache: Path) -> dict[str, object]:
    path = cache / MANIFEST_FILENAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid teacher temporal cache manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("teacher temporal cache manifest must be a JSON object")
    return cast(dict[str, object], manifest)


def _clips(manifest: dict[str, object]) -> list[dict[str, object]]:
    clips = manifest.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("teacher temporal cache manifest.clips: must be a non-empty list")
    return [cast(dict[str, object], clip) for clip in clips]


def _frame_ids(clip: dict[str, object]) -> tuple[int, ...]:
    return tuple(_int_list(clip.get("frame_ids"), "clip.frame_ids"))


def _int_list(value: object, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name}: must be a list")
    output: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"{field_name}[{index}]: must be an integer")
        output.append(item)
    return output


def _required_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name}: must be an integer")
    return value


def _write_json(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "SPLIT_MANIFEST_FILENAME",
    "SPLIT_REPORT_FILENAME",
    "SPLIT_NAMES",
    "build_smgt_tiny_split_manifest",
    "format_smgt_tiny_split_inspection",
    "format_smgt_tiny_split_report",
    "load_smgt_tiny_split_manifest",
    "split_indices",
    "validate_smgt_tiny_split_manifest",
    "write_smgt_tiny_split_manifest",
]
