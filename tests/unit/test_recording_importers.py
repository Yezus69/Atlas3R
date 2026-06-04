import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.forge.clip_cache import (
    CLIP_CACHE_FORMAT_NAME,
    CLIP_CACHE_FORMAT_VERSION,
    CLIP_CACHE_MANIFEST_FILENAME,
    write_clip_payload,
    write_json_file,
)
from atlas3r.recording.importers import (
    SENSOR_CAPTURE_FORMAT_NAME,
    SENSOR_CAPTURE_FORMAT_VERSION,
    ClipCacheRecordingImportConfig,
    SensorFolderRecordingImportConfig,
    TumRecordingImportConfig,
    recording_from_clip_cache,
    recording_from_sensor_folder,
    recording_from_tum_manifest,
)
from atlas3r.recording.schema import load_recording, validate_recording_folder


class RecordingImportersTest(unittest.TestCase):
    def test_from_sensor_folder_writes_valid_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "sensor_capture"
            _write_sensor_capture(capture)
            output = root / "recording"

            result = recording_from_sensor_folder(
                SensorFolderRecordingImportConfig(input=capture, output=output)
            )
            recording = load_recording(output)
            validation = validate_recording_folder(output)

        self.assertEqual(result["frame_count"], 2)
        self.assertEqual(len(recording.frames), 2)
        self.assertTrue(validation["depth_present"])
        self.assertTrue(validation["pose_present"])
        self.assertIn("sensor_capture_root", recording.manifest["external_roots"])  # type: ignore[operator]

    def test_from_sensor_folder_rejects_unsafe_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "sensor_capture"
            _write_sensor_capture(capture)
            lines = (capture / "frames.jsonl").read_text(encoding="utf-8").splitlines()
            first = json.loads(lines[0])
            first["rgb_path"] = "../escape.npz"
            lines[0] = json.dumps(first)
            (capture / "frames.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unsafe path"):
                recording_from_sensor_folder(
                    SensorFolderRecordingImportConfig(input=capture, output=root / "recording")
                )

    def test_from_sensor_folder_allows_missing_depth_and_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "sensor_capture"
            _write_sensor_capture(capture, depth_present=False, pose_present=False)
            output = root / "recording"

            result = recording_from_sensor_folder(
                SensorFolderRecordingImportConfig(input=capture, output=output)
            )
            recording = load_recording(output)

        self.assertFalse(result["depth_present"])
        self.assertFalse(result["pose_present"])
        self.assertIsNone(recording.frames[0].depth_path)
        self.assertIsNone(recording.frames[0].T_world_camera)

    def test_from_clip_cache_writes_valid_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            _write_clip_cache(cache)
            output = root / "recording"

            result = recording_from_clip_cache(
                ClipCacheRecordingImportConfig(
                    clip_cache=cache,
                    output=output,
                    dedupe_frame_id=True,
                )
            )
            recording = load_recording(output)
            depth_exists = (output / "depth" / "frame_000001.npz").is_file()

        self.assertEqual(result["frame_count"], 3)
        self.assertEqual(len(recording.frames), 3)
        self.assertTrue(depth_exists)

    def test_from_tum_manifest_references_external_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tum_root = root / "tum"
            tum_root.mkdir()
            (tum_root / "rgb").mkdir()
            (tum_root / "depth").mkdir()
            (tum_root / "rgb" / "0.png").write_bytes(b"not-used-by-import")
            (tum_root / "depth" / "0.png").write_bytes(b"not-used-by-import")
            manifest_path = root / "tum_manifest.json"
            manifest_path.write_text(json.dumps(_tum_manifest(tum_root)), encoding="utf-8")
            output = root / "recording"

            result = recording_from_tum_manifest(
                TumRecordingImportConfig(
                    manifest=manifest_path,
                    output=output,
                    split="val",
                    max_frames=1,
                    width=4,
                    height=3,
                )
            )
            recording = load_recording(output)

        self.assertEqual(result["frame_count"], 1)
        self.assertEqual(recording.frames[0].depth_scale, 5000.0)
        self.assertIn("tum_root", recording.manifest["external_roots"])  # type: ignore[operator]


def _write_clip_cache(cache: Path) -> None:
    payload = _clip_payload()
    write_clip_payload(
        cache / "clips" / "clip_000000.npz", payload, clip_length=3, height=4, width=4
    )
    manifest = {
        "clip_count": 1,
        "clip_length": 3,
        "clips": [
            {
                "center_index": 1,
                "clip_id": 0,
                "frame_ids": [0, 1, 2],
                "payload_path": "clips/clip_000000.npz",
                "timestamps_s": [0.0, 0.033, 0.066],
            }
        ],
        "format_name": CLIP_CACHE_FORMAT_NAME,
        "format_version": CLIP_CACHE_FORMAT_VERSION,
        "image_height": 4,
        "image_width": 4,
        "max_frame_gap_s": 0.12,
        "source_dataset_name": "unit",
        "source_manifest_path": "unit_manifest.json",
        "source_sequence_name": "unit_sequence",
        "split": "val",
        "stride": 1,
        "teacher_source_metadata": {"depth_source": "unit"},
        "truth_boundary": {
            "accuracy_report": False,
            "diagnostic_only": True,
            "performance_report": False,
            "teacher_source": "tum_rgbd_sensor_depth_pose",
        },
    }
    write_json_file(cache / CLIP_CACHE_MANIFEST_FILENAME, manifest)


def _write_sensor_capture(
    root: Path,
    *,
    depth_present: bool = True,
    pose_present: bool = True,
) -> None:
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    width = 4
    height = 4
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.ones((height, width), dtype=np.float32)
    valid = np.ones((height, width), dtype=np.bool_)
    frames = []
    K = [[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]]
    for frame_id in (0, 1):
        np.savez(root / "rgb" / f"frame_{frame_id:06d}.npz", rgb_u8=rgb)
        record: dict[str, object] = {
            "K": K,
            "frame_id": frame_id,
            "rgb_path": f"rgb/frame_{frame_id:06d}.npz",
            "source_metadata": {"fixture": True},
            "timestamp_s": frame_id * 0.033,
        }
        if depth_present:
            np.savez(
                root / "depth" / f"frame_{frame_id:06d}.npz",
                depth_m=depth,
                valid_depth_mask=valid,
            )
            record["depth_path"] = f"depth/frame_{frame_id:06d}.npz"
        if pose_present:
            transform = np.eye(4, dtype=np.float32)
            transform[0, 3] = np.float32(frame_id * 0.05)
            record["T_world_camera"] = transform.tolist()
            record["camera_center_world_m"] = transform[:3, 3].tolist()
        frames.append(record)
    manifest = {
        "calibration_metadata": {
            "calibration_source": "unit_fixture",
            "depth_source": "unit_npz_depth",
            "depth_units": "meters",
            "intrinsics_source": "unit_fixture",
            "metric_scale_source": "unit_fixture",
            "pose_source": "unit_fixture" if pose_present else "not_available",
        },
        "coordinate_frame": "x_right_y_down_z_forward",
        "depth_present": depth_present,
        "format_name": SENSOR_CAPTURE_FORMAT_NAME,
        "format_version": SENSOR_CAPTURE_FORMAT_VERSION,
        "height": height,
        "pose_present": pose_present,
        "source_dataset": "unit_sensor",
        "source_sequence": "fixture_sequence",
        "truth_boundary": {
            "accuracy_report": False,
            "diagnostic_only": True,
            "hidden_geometry_measured": False,
            "performance_report": False,
            "realtime_claim": False,
        },
        "width": width,
    }
    (root / "sensor_capture.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (root / "frames.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for frame in frames:
            handle.write(json.dumps(frame, sort_keys=True))
            handle.write("\n")


def _clip_payload() -> dict[str, np.ndarray]:
    K = np.array([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    return {
        "K": np.repeat(K[np.newaxis, :, :], 3, axis=0),
        "T_world_camera": np.repeat(np.eye(4, dtype=np.float32)[np.newaxis, :, :], 3, axis=0),
        "center_index": np.array(1, dtype=np.int32),
        "depth_m": np.ones((3, 4, 4), dtype=np.float32),
        "frame_ids": np.array([0, 1, 2], dtype=np.int32),
        "images_rgb_u8": np.zeros((3, 4, 4, 3), dtype=np.uint8),
        "timestamps_s": np.array([0.0, 0.033, 0.066], dtype=np.float64),
        "valid_depth_mask": np.ones((3, 4, 4), dtype=np.bool_),
    }


def _tum_manifest(root: Path) -> dict[str, object]:
    return {
        "coordinate_frame": "x_right_y_down_z_forward",
        "dataset_name": "TUM RGB-D unit",
        "depth": {"scale": 5000.0},
        "format_name": "atlas3r_tum_rgbd_manifest",
        "format_version": 1,
        "frame_count": 1,
        "frames": [
            {
                "T_world_camera": np.eye(4, dtype=np.float32).tolist(),
                "depth_path": "depth/0.png",
                "depth_timestamp_s": 0.0,
                "frame_id": 0,
                "pose_timestamp_s": 0.0,
                "rgb_path": "rgb/0.png",
                "rgb_timestamp_s": 0.0,
                "split": "val",
            }
        ],
        "image": {"height": 480, "width": 640},
        "intrinsics": {
            "K": [[525.0, 0.0, 319.5], [0.0, 525.0, 239.5], [0.0, 0.0, 1.0]],
            "source": "unit",
        },
        "root_path": str(root),
        "sequence_name": "unit_sequence",
    }


if __name__ == "__main__":
    unittest.main()
