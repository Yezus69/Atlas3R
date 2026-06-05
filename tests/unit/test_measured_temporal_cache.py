import json
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
from atlas3r.training.measured_temporal_cache import (
    MANIFEST_FILENAME,
    MeasuredTemporalCacheBuildConfig,
    MeasuredTemporalCacheDataset,
    build_measured_temporal_cache,
    inspect_measured_temporal_cache,
)


class MeasuredTemporalCacheTest(unittest.TestCase):
    def test_cache_builds_shapes_truth_flags_and_inspector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = _write_recording(root / "recording", frame_count=4)
            report = build_measured_temporal_cache(
                MeasuredTemporalCacheBuildConfig(
                    recording=recording,
                    output=root / "cache",
                    clip_length=2,
                    clip_stride=1,
                    image_size=(4, 6),
                )
            )
            manifest = json.loads((root / "cache" / MANIFEST_FILENAME).read_text("utf-8"))
            with np.load(
                root / "cache" / "clips" / "clip_000000.npz", allow_pickle=False
            ) as payload:
                rgb = np.asarray(payload["rgb_u8"])
                depth = np.asarray(payload["depth_m"])
                K = np.asarray(payload["K"])
                target_source = str(np.asarray(payload["target_source"]).item())
            inspection = inspect_measured_temporal_cache(root / "cache")
            dataset = MeasuredTemporalCacheDataset(root / "cache")
            sample = dataset[0]

        self.assertTrue(report["validation_passed"])
        self.assertEqual(manifest["format_name"], "atlas3r_measured_temporal_cache")
        self.assertEqual(manifest["clip_count"], 3)
        self.assertEqual(rgb.shape, (2, 4, 6, 3))
        self.assertEqual(depth.shape, (2, 4, 6))
        self.assertEqual(K.shape, (2, 3, 3))
        self.assertEqual(target_source, "measured")
        self.assertEqual(inspection["clip_count"], 3)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(sample["target_source"], "measured")
        self.assertEqual(sample["depth_target_weight"], 1.0)
        self.assertEqual(sample["pose_target_weight"], 1.0)
        truth = sample["truth_flags"]  # type: ignore[index]
        self.assertTrue(truth["measured_depth_used_for_training"])
        self.assertTrue(truth["measured_pose_used_for_training"])
        self.assertFalse(truth["measured_depth_used"])
        self.assertFalse(truth["measured_pose_used"])
        self.assertFalse(truth["teacher_geometry_used"])

    def test_cache_rejects_unsafe_clip_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = _write_recording(root / "recording", frame_count=2)
            build_measured_temporal_cache(
                MeasuredTemporalCacheBuildConfig(
                    recording=recording,
                    output=root / "cache",
                    clip_length=2,
                    clip_stride=1,
                    image_size=(4, 6),
                )
            )
            manifest_path = root / "cache" / MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_text("utf-8"))
            manifest["clips"][0]["path"] = "../outside.npz"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "safe relative path"):
                inspect_measured_temporal_cache(root / "cache")

    def test_cache_rejects_invalid_depth_and_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = _write_recording(root / "recording", frame_count=2)
            build_measured_temporal_cache(
                MeasuredTemporalCacheBuildConfig(
                    recording=recording,
                    output=root / "cache",
                    clip_length=2,
                    clip_stride=1,
                    image_size=(4, 6),
                )
            )
            clip_path = root / "cache" / "clips" / "clip_000000.npz"
            with np.load(clip_path, allow_pickle=False) as payload:
                data = {key: np.asarray(payload[key]) for key in payload.files}
            data["depth_m"][0, 0, 0] = np.float32(-1.0)
            np.savez_compressed(clip_path, **data)
            with self.assertRaisesRegex(ValueError, "non-negative"):
                inspect_measured_temporal_cache(root / "cache")

            data["depth_m"][0, 0, 0] = np.float32(1.0)
            data["T_world_camera"][0, 3, 3] = np.float32(2.0)
            np.savez_compressed(clip_path, **data)
            with self.assertRaisesRegex(ValueError, "bottom row"):
                inspect_measured_temporal_cache(root / "cache")


def _write_recording(path: Path, *, frame_count: int) -> Path:
    (path / "rgb").mkdir(parents=True)
    (path / "depth").mkdir()
    K = np.asarray([[6.0, 0.0, 2.5], [0.0, 6.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    frames = []
    for frame_id in range(frame_count):
        rgb = np.full((4, 6, 3), frame_id * 20, dtype=np.uint8)
        depth = np.full((4, 6), 1.0 + 0.05 * frame_id, dtype=np.float32)
        valid = np.ones((4, 6), dtype=np.bool_)
        np.savez_compressed(path / "rgb" / f"frame_{frame_id:06d}.npz", rgb_u8=rgb)
        np.savez_compressed(
            path / "depth" / f"frame_{frame_id:06d}.npz",
            depth_m=depth,
            valid_depth_mask=valid,
        )
        T_world_camera = np.eye(4, dtype=np.float32)
        T_world_camera[0, 3] = np.float32(0.05 * frame_id)
        frames.append(
            {
                "frame_id": frame_id,
                "timestamp_s": float(frame_id) * 0.1,
                "rgb_path": f"rgb/frame_{frame_id:06d}.npz",
                "depth_path": f"depth/frame_{frame_id:06d}.npz",
                "K": K.tolist(),
                "T_world_camera": T_world_camera.tolist(),
                "camera_center_world_m": T_world_camera[:3, 3].tolist(),
                "source_metadata": {"fixture": True},
            }
        )
    manifest = {
        "format_name": RECORDING_FORMAT_NAME,
        "format_version": RECORDING_FORMAT_VERSION,
        "coordinate_frame": RECORDING_COORDINATE_FRAME,
        "frame_count": frame_count,
        "width": 6,
        "height": 4,
        "source_dataset": "unit_test",
        "source_sequence": "tiny_measured",
        "capture_metadata": {"fixture": True},
        "known_calibration_metadata": {"K": K.tolist()},
        "depth_present": True,
        "pose_present": True,
        "truth_boundary": recording_truth_boundary(depth_present=True, pose_present=True),
    }
    write_recording_files(path, manifest=manifest, frames=frames)
    return path


if __name__ == "__main__":
    unittest.main()
