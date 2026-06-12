from __future__ import annotations

from atlas3r.metric_council import (
    CAMERA_HEIGHT_FLOOR_PRIOR_ID,
    _camera_height_floor_prior_anchor,
    _consensus,
)


def _pose(frame_id: int, xyz: tuple[float, float, float]) -> dict:
    x, y, z = xyz
    return {
        "frame_id": frame_id,
        "T_world_camera": [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def test_camera_height_prior_uses_registered_pose_median_and_iqr():
    floor = {
        "up_alignment_applied": True,
        "method": "ransac",
        "inlier_ratio": 0.40,
        "floor_axis": 1,
        "floor_value": 0.0,
        "normal": [0.0, 1.0, 0.0],
    }
    rows = [
        _pose(10, (0.0, 1.00, 0.0)),
        _pose(11, (0.0, 1.35, 0.0)),
        _pose(12, (0.0, 1.70, 0.0)),
    ]

    report = _camera_height_floor_prior_anchor("synthetic", floor, rows)

    assert report["status"] == "weak"
    assert report["verdict"] == "weak"
    assert report["estimated_metric_scale"] == 1.0
    assert report["relative_scale_uncertainty"] == 0.20
    stat = report["height_statistic"]
    assert stat["registered_pose_count"] == 3
    assert stat["frame_ids_used"] == (10, 11, 12)
    assert stat["height_median_reconstruction_units"] == 1.35
    assert round(stat["height_iqr_reconstruction_units"], 6) == 0.35


def test_camera_height_prior_refuses_when_floor_not_aligned():
    floor = {
        "up_alignment_applied": False,
        "method": "ransac",
        "inlier_ratio": 0.10,
        "floor_axis": 2,
        "floor_value": -0.2,
        "normal": [0.0, 0.0, 1.0],
    }

    report = _camera_height_floor_prior_anchor(
        "synthetic", floor, [_pose(0, (0.0, 0.0, 1.0)), _pose(1, (0.0, 0.0, 1.2))]
    )

    assert report["status"] == "unavailable"
    assert report["verdict"] == "unavailable"
    assert "estimated_metric_scale" not in report
    assert report["blockers"] == ("camera_height_floor_prior_floor_unavailable:none",)


def test_camera_height_prior_is_guarded_to_reportage_only():
    floor = {
        "up_alignment_applied": True,
        "method": "ransac",
        "inlier_ratio": 0.40,
        "floor_axis": 2,
        "floor_value": 0.0,
        "normal": [0.0, 0.0, 1.0],
    }
    rows = [_pose(0, (0.0, 0.0, 0.675)), _pose(1, (0.0, 0.0, 0.675))]
    anchor = _camera_height_floor_prior_anchor("synthetic", floor, rows)

    consensus = _consensus([anchor])

    assert anchor["estimated_metric_scale"] == 2.0
    assert consensus["status"] == "uncalibrated_anchor_reportage_only"
    assert consensus["scale_mean"] == 1.0
    assert consensus["relative_scale_uncertainty"] == 0.30
    assert consensus["weak_anchor_ids"] == (CAMERA_HEIGHT_FLOOR_PRIOR_ID,)
    assert consensus["raw_uncalibrated_consensus"]["scale_mean"] == 2.0
