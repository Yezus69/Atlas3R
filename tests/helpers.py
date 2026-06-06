from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def write_ppm(path: Path, rgb: NDArray[np.uint8]) -> None:
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("expected RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes())


def write_ppm_sequence(root: Path, *, count: int = 4, width: int = 4, height: int = 3) -> None:
    root.mkdir(parents=True, exist_ok=True)
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    for index in range(count):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = (x_grid * 30 + index * 20).astype(np.uint8)
        rgb[:, :, 1] = (y_grid * 40 + index * 10).astype(np.uint8)
        rgb[:, :, 2] = np.uint8(50 + index * 20)
        write_ppm(root / f"{index:03d}.ppm", rgb)


def write_fake_vggt_cache(
    root: Path,
    *,
    frame_ids: tuple[int, ...] = (0, 1),
    width: int = 4,
    height: int = 3,
    depth_m: float = 2.0,
) -> Path:
    proposals = root / "proposals"
    proposals.mkdir(parents=True, exist_ok=True)
    truth = {
        "label_type": "teacher_pseudo",
        "metric_scale_source": "vggt_unanchored_metric_proposal",
        "measured_geometry": False,
        "observed_only": True,
        "predicted_completion": False,
        "hidden_geometry_measured": False,
        "accuracy_report": False,
        "realtime_claim": False,
        "usable_for_training": False,
        "notes": "fake VGGT cache fixture",
    }
    K = [
        [float(max(width, height)), 0.0, float(width - 1) / 2.0],
        [0.0, float(max(width, height)), float(height - 1) / 2.0],
        [0.0, 0.0, 1.0],
    ]
    camera_rows = []
    depth_records = []
    arrays: dict[str, NDArray[np.float32]] = {}
    for index, frame_id in enumerate(frame_ids):
        T_world_camera = np.eye(4, dtype=np.float32)
        T_world_camera[0, 3] = float(index) * 0.1
        depth_key = f"vggt_depth_{index:06d}"
        sigma_key = f"vggt_depth_sigma_{index:06d}"
        confidence_key = f"vggt_confidence_{index:06d}"
        valid_key = f"vggt_valid_mask_{index:06d}"
        arrays[depth_key] = np.full((height, width), depth_m, dtype=np.float32)
        arrays[sigma_key] = np.full((height, width), 0.25, dtype=np.float32)
        arrays[confidence_key] = np.full((height, width), 0.8, dtype=np.float32)
        arrays[valid_key] = np.ones((height, width), dtype=np.float32)
        camera_rows.append(
            {
                "teacher_name": "vggt",
                "frame_id": frame_id,
                "keyframe_index": index,
                "window_id": 0,
                "K": K,
                "T_world_camera": T_world_camera.tolist(),
                "camera_center_world_m": T_world_camera[:3, 3].tolist(),
                "pose_confidence": 0.8,
                "intrinsics_source": "vggt",
                "pose_source": "vggt",
                "metric_scale_source": "vggt_unanchored_metric_proposal",
                "coordinate_convention": "x_right_y_down_z_forward",
                "truth_boundary": truth,
                "depth_key": depth_key,
                "depth_sigma_key": sigma_key,
                "confidence_key": confidence_key,
                "valid_mask_key": valid_key,
                "depth_shape": [height, width],
                "pseudo_submap_id": 0,
                "depth_scale_applied": 1.0,
                "stitch_status": "fixture",
            }
        )
        depth_records.append(
            {
                "teacher_name": "vggt",
                "frame_id": frame_id,
                "keyframe_index": index,
                "window_id": 0,
                "depth_key": depth_key,
                "depth_sigma_key": sigma_key,
                "confidence_key": confidence_key,
                "valid_mask_key": valid_key,
                "depth_shape": [height, width],
                "depth_source": "vggt",
                "metric_scale_source": "vggt_unanchored_metric_proposal",
                "coordinate_convention": "x_right_y_down_z_forward",
                "truth_boundary": truth,
                "confidence_derived": False,
            }
        )
    _write_jsonl(proposals / "vggt_cameras.jsonl", camera_rows)
    _write_jsonl(
        proposals / "vggt_windows.jsonl",
        [
            {
                "teacher_name": "vggt",
                "window_id": 0,
                "frame_ids": list(frame_ids),
                "status": "fixture",
                "stitch_mode": "overlap-sim3",
                "stitch_status": "accepted_or_root",
                "truth_boundary": truth,
            }
        ],
    )
    metadata = {
        "teacher_name": "vggt",
        "depth_records": depth_records,
        "vggt_metadata": {
            "teacher_name": "vggt",
            "runtime_ms": 0.0,
            "stitching": {
                "vggt_window_count": 1,
                "stitch_mode": "overlap-sim3",
                "stitch_edge_count": 0,
                "accepted_edge_count": 0,
                "rejected_edge_count": 0,
                "pseudo_submap_count": 1,
                "overlap_center_rmse_m": None,
                "scale_min": 1.0,
                "scale_median": 1.0,
                "scale_max": 1.0,
                "rejected_windows": [],
            },
        },
        "truth_boundary": truth,
    }
    np.savez_compressed(
        proposals / "vggt_depths.npz",
        **arrays,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    (proposals / "proposal_manifest.json").write_text(
        json.dumps(
            {
                "format_name": "atlas3r_teacher_proposal_cache",
                "format_version": 1,
                "teacher_counts": {
                    "vggt_camera_proposals": len(camera_rows),
                    "vggt_depth_proposals": len(depth_records),
                },
                "truth_boundary": truth,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return proposals


def write_fake_depth_pro_cache(
    root: Path,
    *,
    frame_ids: tuple[int, ...] = (0, 1),
    width: int = 4,
    height: int = 3,
    depth_m: float = 2.0,
    focal_px: float | None = 4.0,
) -> Path:
    proposals = root / "proposals"
    proposals.mkdir(parents=True, exist_ok=True)
    truth = {
        "label_type": "teacher_pseudo",
        "metric_scale_source": "depth_pro_metric_proposal_unanchored",
        "measured_geometry": False,
        "observed_only": True,
        "predicted_completion": False,
        "hidden_geometry_measured": False,
        "accuracy_report": False,
        "realtime_claim": False,
        "usable_for_training": False,
        "notes": "fake Depth Pro cache fixture",
    }
    K = (
        None
        if focal_px is None
        else [
            [float(focal_px), 0.0, float(width - 1) / 2.0],
            [0.0, float(focal_px), float(height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    camera_rows = []
    depth_records = []
    frame_rows = []
    arrays: dict[str, NDArray[np.float32]] = {}
    for index, frame_id in enumerate(frame_ids):
        depth_key = f"depth_pro_depth_{index:06d}"
        sigma_key = f"depth_pro_depth_sigma_{index:06d}"
        confidence_key = f"depth_pro_confidence_{index:06d}"
        valid_key = f"depth_pro_valid_mask_{index:06d}"
        arrays[depth_key] = np.full((height, width), depth_m, dtype=np.float32)
        arrays[sigma_key] = np.full((height, width), 0.2, dtype=np.float32)
        arrays[confidence_key] = np.full((height, width), 0.7, dtype=np.float32)
        arrays[valid_key] = np.ones((height, width), dtype=np.float32)
        camera_rows.append(
            {
                "teacher_name": "depth_pro",
                "frame_id": frame_id,
                "keyframe_index": index,
                "K": K,
                "focal_px": focal_px,
                "fx": focal_px,
                "fy": focal_px,
                "intrinsics_source": "depth_pro" if focal_px is not None else "unavailable",
                "metric_scale_source": "depth_pro_metric_proposal_unanchored",
                "coordinate_convention": "x_right_y_down_z_forward",
                "truth_boundary": truth,
                "depth_key": depth_key,
                "depth_shape": [height, width],
            }
        )
        depth_records.append(
            {
                "teacher_name": "depth_pro",
                "frame_id": frame_id,
                "keyframe_index": index,
                "depth_key": depth_key,
                "depth_sigma_key": sigma_key,
                "confidence_key": confidence_key,
                "valid_mask_key": valid_key,
                "depth_shape": [height, width],
                "depth_source": "depth_pro",
                "metric_scale_source": "depth_pro_metric_proposal_unanchored",
                "coordinate_convention": "x_right_y_down_z_forward",
                "truth_boundary": truth,
                "confidence_derived": True,
            }
        )
        frame_rows.append(
            {
                "teacher_name": "depth_pro",
                "frame_id": frame_id,
                "keyframe_index": index,
                "status": "fixture",
                "runtime_ms": 0.0,
                "output_shapes": {"depth": [height, width], "focallength_px": []},
                "truth_boundary": truth,
            }
        )
    _write_jsonl(proposals / "depth_pro_cameras.jsonl", camera_rows)
    _write_jsonl(proposals / "depth_pro_frames.jsonl", frame_rows)
    metadata = {
        "teacher_name": "depth_pro",
        "depth_records": depth_records,
        "depth_pro_metadata": {
            "teacher_name": "depth_pro",
            "model_source": "fixture",
            "checkpoint": "fixture",
            "device": "cpu",
            "image_size": None,
            "frame_ids": list(frame_ids),
            "runtime_ms": 0.0,
            "output_shapes": {
                str(frame_id): {"depth": [height, width], "focallength_px": []}
                for frame_id in frame_ids
            },
            "truth_boundary": truth,
        },
        "truth_boundary": truth,
    }
    np.savez_compressed(
        proposals / "depth_pro_depths.npz",
        **arrays,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    (proposals / "proposal_manifest.json").write_text(
        json.dumps(
            {
                "format_name": "atlas3r_teacher_proposal_cache",
                "format_version": 1,
                "teacher_counts": {
                    "depth_pro_camera_proposals": len(camera_rows),
                    "depth_pro_depth_proposals": len(depth_records),
                },
                "truth_boundary": truth,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return proposals


def write_fake_colmap_text_model(
    root: Path,
    *,
    frame_ids: tuple[int, ...] = (0, 1, 2, 3),
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "cameras.txt").write_text(
        "1 SIMPLE_RADIAL 4 3 4 1.5 1 0\n",
        encoding="utf-8",
    )
    image_lines = []
    point_lines = []
    for index, frame_id in enumerate(frame_ids):
        center_x = float(index) * 0.1
        image_lines.append(f"{index + 1} 1 0 0 0 {-center_x} 0 0 1 frame_{frame_id:06d}.ppm")
        image_lines.append("0 0 1")
        point_lines.append(
            f"{index + 1} {center_x} 0 2 {50 + index} {80 + index} {120 + index} 0.1 {index + 1} 0"
        )
    (root / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (root / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")
    return root


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
