"""Small camera metadata helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from atlas3r.contracts.frames import CameraModel


def camera_from_focal(
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float | None = None,
    cy: float | None = None,
    source: str = "metadata",
    confidence: float = 1.0,
) -> CameraModel:
    K = np.array(
        [
            [fx, 0.0, float(width - 1) / 2.0 if cx is None else cx],
            [0.0, fy, float(height - 1) / 2.0 if cy is None else cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return CameraModel(width=width, height=height, K=K, confidence=confidence, source=source)


def scale_camera_model(camera: CameraModel, *, width: int, height: int) -> CameraModel:
    sx = float(width) / float(camera.width)
    sy = float(height) / float(camera.height)
    K = camera.K.copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return CameraModel(
        width=width,
        height=height,
        K=K,
        distortion_model=camera.distortion_model,
        distortion_params=camera.distortion_params,
        rolling_shutter_row_time_s=camera.rolling_shutter_row_time_s,
        confidence=camera.confidence,
        source=camera.source,
    )


@dataclass(frozen=True)
class ImageMetadata:
    source_uri: str
    width: int | None
    height: int | None
    focal_length_mm: float | None = None
    focal_length_35mm: float | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    timestamp: str | None = None
    orientation: int | None = None
    exif_available: bool = False
    metadata_status: str = "missing"
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_uri": self.source_uri,
            "width": self.width,
            "height": self.height,
            "focal_length_mm": self.focal_length_mm,
            "focal_length_35mm": self.focal_length_35mm,
            "camera_make": self.camera_make,
            "camera_model": self.camera_model,
            "timestamp": self.timestamp,
            "orientation": self.orientation,
            "exif_available": self.exif_available,
            "metadata_status": self.metadata_status,
            "error": self.error,
        }


def read_image_metadata(
    path: str, *, width: int | None = None, height: int | None = None
) -> ImageMetadata:
    """Read optional image metadata without making Pillow an import-time dependency."""
    try:
        from PIL import ExifTags, Image
    except Exception as exc:
        return ImageMetadata(
            source_uri=path,
            width=width,
            height=height,
            metadata_status="pillow_unavailable",
            error=str(exc),
        )
    try:
        with Image.open(path) as image:
            image_width, image_height = image.size
            exif = image.getexif()
            tag_names = {value: key for key, value in ExifTags.TAGS.items()}
            make = _string_or_none(exif.get(tag_names.get("Make", -1)))
            model = _string_or_none(exif.get(tag_names.get("Model", -1)))
            timestamp = _string_or_none(
                exif.get(tag_names.get("DateTimeOriginal", -1))
                or exif.get(tag_names.get("DateTime", -1))
            )
            orientation = _int_or_none(exif.get(tag_names.get("Orientation", -1)))
            focal_mm = _float_or_none(exif.get(tag_names.get("FocalLength", -1)))
            focal_35mm = _float_or_none(exif.get(tag_names.get("FocalLengthIn35mmFilm", -1)))
            has_exif = bool(exif)
            has_useful = any(
                value is not None
                for value in (make, model, timestamp, orientation, focal_mm, focal_35mm)
            )
            return ImageMetadata(
                source_uri=path,
                width=image_width,
                height=image_height,
                focal_length_mm=focal_mm,
                focal_length_35mm=focal_35mm,
                camera_make=make,
                camera_model=model,
                timestamp=timestamp,
                orientation=orientation,
                exif_available=has_exif,
                metadata_status="available" if has_useful else "no_useful_exif",
            )
    except Exception as exc:
        return ImageMetadata(
            source_uri=path,
            width=width,
            height=height,
            metadata_status="read_failed",
            error=str(exc),
        )


def metadata_summary(records: tuple[ImageMetadata, ...]) -> dict[str, object]:
    frame_count = len(records)
    focal_mm = _sorted_unique_float(item.focal_length_mm for item in records)
    focal_35mm = _sorted_unique_float(item.focal_length_35mm for item in records)
    make_models = sorted(
        {
            " ".join(part for part in (item.camera_make, item.camera_model) if part).strip()
            for item in records
            if item.camera_make or item.camera_model
        }
    )
    orientation_values = sorted(
        item.orientation for item in records if item.orientation is not None
    )
    exif_count = sum(1 for item in records if item.exif_available)
    useful_count = sum(1 for item in records if item.metadata_status == "available")
    metadata_quality = (
        "focal_metadata_available"
        if focal_mm or focal_35mm
        else "camera_metadata_available"
        if useful_count > 0
        else "exif_present_without_focal"
        if exif_count > 0
        else "missing"
    )
    return {
        "format_name": "atlas3r_frame_metadata_summary",
        "format_version": 1,
        "frame_count": frame_count,
        "frame_count_with_exif": exif_count,
        "frame_count_with_useful_metadata": useful_count,
        "camera_make_models": make_models,
        "focal_mm_values": focal_mm,
        "focal_35mm_values": focal_35mm,
        "orientation_values": orientation_values,
        "timestamp_count": sum(1 for item in records if item.timestamp is not None),
        "metadata_quality": metadata_quality,
        "intrinsics_proposal_only": True,
        "physical_accuracy_claim": False,
        "records": [item.to_dict() for item in records],
    }


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, tuple) and len(value) == 2:
            denominator = float(str(value[1]))
            return None if denominator == 0.0 else float(str(value[0])) / denominator
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            denominator = float(str(value.denominator))
            return None if denominator == 0.0 else float(str(value.numerator)) / denominator
        return float(str(value))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _int_or_none(value: object) -> int | None:
    try:
        return None if value is None else int(str(value))
    except (TypeError, ValueError):
        return None


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sorted_unique_float(values: Iterable[float | None]) -> list[float]:
    unique: set[float] = set()
    for value in values:
        if value is not None and np.isfinite(float(value)):
            unique.add(round(float(value), 6))
    return sorted(unique)
