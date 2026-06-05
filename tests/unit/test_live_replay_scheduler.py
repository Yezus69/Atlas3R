import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.recording.schema import (
    RECORDING_COORDINATE_FRAME,
    RECORDING_FORMAT_NAME,
    RECORDING_FORMAT_VERSION,
    recording_truth_boundary,
    write_recording_files,
)
from atlas3r.runtime.live_replay import LiveReplayConfig, run_live_replay_recording

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class LiveReplaySchedulerTest(unittest.TestCase):
    def test_live_replay_writes_summary_events_and_sparse_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = root / "recording"
            _write_replay_recording(recording, frame_count=4)
            output = root / "live"

            summary = run_live_replay_recording(
                LiveReplayConfig(
                    recording=recording,
                    output=output,
                    target_fps=30.0,
                    max_frames=4,
                    map_keyframe_stride=1,
                    max_capture_queue=4,
                    max_map_queue=2,
                    voxel_size_m=0.25,
                    truncation_voxels=3.0,
                    export_point_cloud=True,
                )
            )
            events = _jsonl(output / "live_replay_events.jsonl")
            report = (output / "live_replay_report.md").read_text(encoding="utf-8")
            sparse_state_exists = (output / "sparse_tsdf" / "sparse_tsdf_state.npz").is_file()
            sparse_surface_exists = (output / "sparse_tsdf" / "surface_points.npz").is_file()
            point_cloud_exists = (output / "surface_points.ply").is_file()
            summary_json_exists = (output / "live_replay_summary.json").is_file()

        self.assertEqual(summary["format_name"], "atlas3r_live_replay_summary")
        self.assertTrue(summary["simulated_pacing"])
        self.assertEqual(summary["target_fps"], 30.0)
        self.assertEqual(summary["frame_count_seen"], 4)
        self.assertEqual(summary["frame_count_emitted"], 4)
        self.assertEqual(summary["pose_update_count"], 4)
        self.assertEqual(summary["keyframe_selected_count"], 4)
        self.assertEqual(summary["map_update_count"], 4)
        self.assertEqual(summary["mapper_backend"], "cpu-sparse")
        self.assertTrue(sparse_state_exists)
        self.assertTrue(sparse_surface_exists)
        self.assertTrue(point_cloud_exists)
        self.assertTrue(summary_json_exists)
        self.assertEqual([event["event_index"] for event in events], list(range(len(events))))
        self.assertFalse(summary["truth_boundary"]["accuracy_report"])
        self.assertFalse(summary["truth_boundary"]["realtime_claim"])
        self.assertFalse(summary["truth_boundary"]["rgb_only_mapping_ready"])
        self.assertFalse(summary["truth_boundary"]["hidden_geometry_measured"])
        self.assertTrue(summary["truth_boundary"]["measured_depth_used"])
        self.assertTrue(summary["truth_boundary"]["measured_pose_used"])
        self.assertIn("not live-mapping-ready evidence", report)
        self.assertIn("no RGB-only student mapping", report)

    def test_bounded_queues_drop_with_explicit_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = root / "recording"
            _write_replay_recording(recording, frame_count=5)
            output = root / "live"

            summary = run_live_replay_recording(
                LiveReplayConfig(
                    recording=recording,
                    output=output,
                    max_frames=5,
                    map_keyframe_stride=1,
                    max_capture_queue=2,
                    max_map_queue=1,
                    voxel_size_m=0.25,
                    truncation_voxels=3.0,
                    capture_service_interval_frames=99,
                    map_service_interval_frames=99,
                )
            )
            events = _jsonl(output / "live_replay_events.jsonl")
            reasons = {event["drop_reason"] for event in events if event["drop_reason"]}

        self.assertLessEqual(summary["max_capture_queue_depth_observed"], 2)
        self.assertLessEqual(summary["max_map_queue_depth_observed"], 1)
        self.assertEqual(summary["frame_count_seen"], 5)
        self.assertEqual(summary["frame_count_emitted"], 2)
        self.assertEqual(summary["dropped_frame_count"], 3)
        self.assertEqual(summary["dropped_keyframe_count"], 1)
        self.assertIn("capture_queue_full", reasons)
        self.assertIn("map_queue_full", reasons)

    def test_stride_keyframe_decisions_include_not_keyframe_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = root / "recording"
            _write_replay_recording(recording, frame_count=4)
            output = root / "live"

            summary = run_live_replay_recording(
                LiveReplayConfig(
                    recording=recording,
                    output=output,
                    max_frames=4,
                    map_keyframe_stride=2,
                    voxel_size_m=0.25,
                    truncation_voxels=3.0,
                )
            )
            events = _jsonl(output / "live_replay_events.jsonl")
            decision_reasons = [
                event["keyframe_reason"]
                for event in events
                if event["stage_name"] == "keyframe_decision"
            ]
            drop_reasons = [event["drop_reason"] for event in events if event["drop_reason"]]

        self.assertEqual(summary["keyframe_selected_count"], 2)
        self.assertEqual(summary["map_update_count"], 2)
        self.assertIn("first_frame", decision_reasons)
        self.assertIn("stride", decision_reasons)
        self.assertIn("not_selected", decision_reasons)
        self.assertIn("not_keyframe", drop_reasons)

    def test_missing_depth_or_pose_skips_mapping_without_fake_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_depth_recording = root / "missing_depth"
            _write_replay_recording(missing_depth_recording, frame_count=2, depth_present=False)
            missing_pose_recording = root / "missing_pose"
            _write_replay_recording(missing_pose_recording, frame_count=2, pose_present=False)

            missing_depth_summary = run_live_replay_recording(
                LiveReplayConfig(
                    recording=missing_depth_recording,
                    output=root / "missing_depth_run",
                    max_frames=2,
                    voxel_size_m=0.25,
                    truncation_voxels=3.0,
                )
            )
            missing_pose_summary = run_live_replay_recording(
                LiveReplayConfig(
                    recording=missing_pose_recording,
                    output=root / "missing_pose_run",
                    max_frames=2,
                    voxel_size_m=0.25,
                    truncation_voxels=3.0,
                )
            )
            depth_reasons = {
                event["drop_reason"]
                for event in _jsonl(root / "missing_depth_run" / "live_replay_events.jsonl")
                if event["drop_reason"]
            }
            pose_reasons = {
                event["drop_reason"]
                for event in _jsonl(root / "missing_pose_run" / "live_replay_events.jsonl")
                if event["drop_reason"]
            }

        self.assertEqual(missing_depth_summary["map_update_count"], 0)
        self.assertEqual(missing_depth_summary["surface_point_count"], 0)
        self.assertTrue(missing_depth_summary["truth_boundary"]["measured_pose_used"])
        self.assertFalse(missing_depth_summary["truth_boundary"]["measured_depth_used"])
        self.assertIn("missing_depth", depth_reasons)
        self.assertEqual(missing_pose_summary["pose_update_count"], 0)
        self.assertEqual(missing_pose_summary["map_update_count"], 0)
        self.assertFalse(missing_pose_summary["truth_boundary"]["measured_pose_used"])
        self.assertFalse(missing_pose_summary["truth_boundary"]["measured_depth_used"])
        self.assertIn("missing_pose", pose_reasons)

    def test_live_replay_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = root / "recording"
            output = root / "live"
            _write_replay_recording(recording, frame_count=5)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "runtime",
                    "live-replay-recording",
                    "--recording",
                    str(recording),
                    "--output",
                    str(output),
                    "--target-fps",
                    "30",
                    "--max-frames",
                    "5",
                    "--mapper-backend",
                    "cpu-sparse",
                    "--map-keyframe-stride",
                    "1",
                    "--max-capture-queue",
                    "4",
                    "--max-map-queue",
                    "2",
                    "--drop-policy",
                    "oldest",
                    "--voxel-size-m",
                    "0.25",
                    "--truncation-voxels",
                    "3.0",
                    "--export-point-cloud",
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            cli_summary = json.loads(
                (output / "live_replay_summary.json").read_text(encoding="utf-8")
            )
            events_exists = (output / "live_replay_events.jsonl").is_file()
            report_exists = (output / "live_replay_report.md").is_file()
            sparse_state_exists = (output / "sparse_tsdf" / "sparse_tsdf_state.npz").is_file()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(cli_summary["frame_count_seen"], 5)
        self.assertTrue(events_exists)
        self.assertTrue(report_exists)
        self.assertTrue(sparse_state_exists)


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_replay_recording(
    root: Path,
    *,
    frame_count: int,
    depth_present: bool = True,
    pose_present: bool = True,
) -> None:
    (root / "rgb").mkdir(parents=True)
    if depth_present:
        (root / "depth").mkdir(parents=True)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.ones((4, 4), dtype=np.float32)
    valid = np.ones((4, 4), dtype=np.bool_)
    frames = []
    for frame_id in range(frame_count):
        np.savez(root / "rgb" / f"frame_{frame_id:06d}.npz", rgb_u8=rgb)
        frame: dict[str, object] = {
            "K": [[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]],
            "frame_id": frame_id,
            "rgb_path": f"rgb/frame_{frame_id:06d}.npz",
            "source_metadata": {"fixture": True},
            "timestamp_s": frame_id / 30.0,
        }
        if depth_present:
            np.savez(
                root / "depth" / f"frame_{frame_id:06d}.npz",
                depth_m=depth,
                valid_depth_mask=valid,
            )
            frame["depth_path"] = f"depth/frame_{frame_id:06d}.npz"
        if pose_present:
            transform = np.eye(4, dtype=np.float32)
            transform[0, 3] = np.float32(frame_id * 0.05)
            frame["T_world_camera"] = transform.tolist()
            frame["camera_center_world_m"] = transform[:3, 3].tolist()
        frames.append(frame)
    write_recording_files(
        root,
        manifest={
            "capture_metadata": {"source": "live_replay_unit"},
            "coordinate_frame": RECORDING_COORDINATE_FRAME,
            "depth_present": depth_present,
            "external_roots": {},
            "format_name": RECORDING_FORMAT_NAME,
            "format_version": RECORDING_FORMAT_VERSION,
            "frame_count": len(frames),
            "height": 4,
            "known_calibration_metadata": {"source": "unit"},
            "pose_present": pose_present,
            "source_dataset": "unit",
            "source_sequence": "live_replay_unit",
            "truth_boundary": recording_truth_boundary(
                depth_present=depth_present,
                pose_present=pose_present,
            ),
            "width": 4,
        },
        frames=frames,
    )


if __name__ == "__main__":
    unittest.main()
