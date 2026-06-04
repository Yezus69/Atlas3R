import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.mapping.cpu_tsdf import TSDFSurface, TSDFVolume
from atlas3r.mapping.mesh_extraction import export_tsdf_triangle_mesh, mesh_dependency_available
from atlas3r.recording.schema import (
    RECORDING_COORDINATE_FRAME,
    RECORDING_FORMAT_NAME,
    RECORDING_FORMAT_VERSION,
    recording_truth_boundary,
    write_recording_files,
)
from atlas3r.runtime.recording_fusion import FuseRecordingConfig, run_fuse_recording


class RecordingRuntimeTest(unittest.TestCase):
    def test_fuse_recording_writes_reports_and_point_cloud(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = root / "recording"
            _write_runtime_recording(recording)
            output = root / "run"

            summary = run_fuse_recording(
                FuseRecordingConfig(
                    recording=recording,
                    output=output,
                    max_frames=2,
                    keyframe_stride=1,
                    voxel_size_m=0.25,
                    truncation_voxels=3.0,
                    export_point_cloud=True,
                    export_mesh="off",
                )
            )
            ply_text = (output / "surface_points.ply").read_text(encoding="utf-8")
            runtime_events_exists = (output / "runtime_events.jsonl").is_file()
            latency_report_exists = (output / "latency_report.json").is_file()

        self.assertEqual(summary["frame_count"], 2)
        self.assertEqual(summary["mode"], "batch")
        self.assertGreater(int(summary["surface_point_count"]), 0)
        self.assertTrue(runtime_events_exists)
        self.assertTrue(latency_report_exists)
        self.assertIn("element vertex", ply_text)
        self.assertIn("property float uncertainty_m", ply_text)

    def test_incremental_fuse_recording_writes_per_frame_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = root / "recording"
            _write_runtime_recording(recording, frame_count=3)
            output = root / "run"

            summary = run_fuse_recording(
                FuseRecordingConfig(
                    recording=recording,
                    output=output,
                    max_frames=2,
                    keyframe_stride=1,
                    voxel_size_m=0.25,
                    truncation_voxels=3.0,
                    export_point_cloud=True,
                    export_mesh="off",
                    mode="incremental",
                )
            )
            events = [
                json.loads(line)
                for line in (output / "per_frame_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            latency_report = json.loads(
                (output / "latency_report.json").read_text(encoding="utf-8")
            )
            surface_points_exists = (output / "surface_points.ply").is_file()

        self.assertEqual(summary["mode"], "incremental")
        self.assertEqual(summary["frame_count"], 2)
        self.assertEqual(len(events), 2)
        self.assertTrue(all(event["selected_keyframe"] for event in events))
        self.assertIn("per_frame_observation_load", latency_report["segments"])
        self.assertIn("per_frame_map_update", latency_report["segments"])
        self.assertTrue(surface_points_exists)

    def test_mesh_auto_records_missing_optional_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = export_tsdf_triangle_mesh(
                root,
                volume=_dummy_volume(),
                surface=_dummy_surface(),
                export_mode="auto",
                force_dependency_missing=True,
            )

        self.assertFalse(status["mesh_exported"])
        self.assertIn("scikit-image", status["reason"])
        self.assertIn("install_hint", status)

    def test_mesh_required_exports_when_dependency_available_otherwise_records_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            if mesh_dependency_available():
                status = export_tsdf_triangle_mesh(
                    root,
                    volume=_crossing_volume(),
                    surface=_dummy_surface(),
                    export_mode="required",
                )
                self.assertTrue(status["mesh_exported"])
                self.assertGreater(int(status["vertex_count"]), 0)
                self.assertGreater(int(status["face_count"]), 0)
                self.assertTrue((root / "mesh.obj").is_file())
            else:
                status = export_tsdf_triangle_mesh(
                    root,
                    volume=_dummy_volume(),
                    surface=_dummy_surface(),
                    export_mode="auto",
                    force_dependency_missing=True,
                )
                self.assertFalse(status["mesh_exported"])
                self.assertIn("install_hint", status)


def _write_runtime_recording(root: Path, *, frame_count: int = 2) -> None:
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    valid = np.ones((4, 4), dtype=np.bool_)
    depth = np.ones((4, 4), dtype=np.float32)
    frames = []
    for frame_id in range(frame_count):
        np.savez(root / "rgb" / f"frame_{frame_id:06d}.npz", rgb_u8=rgb)
        np.savez(
            root / "depth" / f"frame_{frame_id:06d}.npz",
            depth_m=depth,
            valid_depth_mask=valid,
        )
        transform = np.eye(4, dtype=np.float32)
        transform[0, 3] = np.float32(frame_id * 0.05)
        frames.append(
            {
                "K": [[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]],
                "T_world_camera": transform.tolist(),
                "camera_center_world_m": transform[:3, 3].tolist(),
                "depth_path": f"depth/frame_{frame_id:06d}.npz",
                "frame_id": frame_id,
                "rgb_path": f"rgb/frame_{frame_id:06d}.npz",
                "source_metadata": {"fixture": True},
                "timestamp_s": frame_id * 0.033,
            }
        )
    manifest = {
        "capture_metadata": {"source": "runtime_unit"},
        "coordinate_frame": RECORDING_COORDINATE_FRAME,
        "depth_present": True,
        "external_roots": {},
        "format_name": RECORDING_FORMAT_NAME,
        "format_version": RECORDING_FORMAT_VERSION,
        "frame_count": len(frames),
        "height": 4,
        "known_calibration_metadata": {"source": "unit"},
        "pose_present": True,
        "source_dataset": "unit",
        "source_sequence": "runtime_unit",
        "truth_boundary": recording_truth_boundary(depth_present=True, pose_present=True),
        "width": 4,
    }
    write_recording_files(root, manifest=manifest, frames=frames)


def _dummy_volume() -> TSDFVolume:
    return TSDFVolume(
        grid_min_corner_world_m=np.zeros(3, dtype=np.float32),
        voxel_size_m=0.1,
        truncation_distance_m=0.3,
        tsdf=np.ones((2, 2, 2), dtype=np.float32),
        weight=np.ones((2, 2, 2), dtype=np.float32),
        source_frame_ids=(0,),
        coordinate_frame="x_right_y_down_z_forward",
        metric_scale_source="external_pose",
    )


def _crossing_volume() -> TSDFVolume:
    tsdf = np.ones((3, 3, 3), dtype=np.float32)
    tsdf[1:, :, :] = -1.0
    return TSDFVolume(
        grid_min_corner_world_m=np.zeros(3, dtype=np.float32),
        voxel_size_m=0.1,
        truncation_distance_m=0.3,
        tsdf=tsdf,
        weight=np.ones((3, 3, 3), dtype=np.float32),
        source_frame_ids=(0,),
        coordinate_frame="x_right_y_down_z_forward",
        metric_scale_source="external_pose",
    )


def _dummy_surface() -> TSDFSurface:
    return TSDFSurface(
        points_world_m=np.zeros((1, 3), dtype=np.float32),
        confidence=np.ones((1,), dtype=np.float32),
        uncertainty_m=np.full((1,), 0.1, dtype=np.float32),
        voxel_indices_xyz=np.zeros((1, 3), dtype=np.int32),
        metadata={"source_frame_ids": [0], "voxel_size_m": 0.1},
    )


if __name__ == "__main__":
    unittest.main()
