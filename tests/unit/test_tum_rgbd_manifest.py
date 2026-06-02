import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from atlas3r.data.tum_rgbd import prepare_tum_rgbd_manifest, safe_extract_tar


class TumRgbdManifestTest(unittest.TestCase):
    def test_prepare_manifest_associates_sorted_frames_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rgbd_dataset_freiburg1_xyz"
            _write_minimal_tum_sequence(root, frame_count=3)
            output = Path(tmp) / "manifest.json"

            manifest = prepare_tum_rgbd_manifest(
                root,
                output,
                stride=1,
                max_frames=2,
            )

            self.assertTrue(output.is_file())
            on_disk = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["format_name"], "atlas3r_tum_rgbd_manifest")
            self.assertEqual(manifest["frame_count"], 2)
            self.assertEqual(manifest["depth"]["scale"], 5000.0)
            self.assertEqual(manifest["intrinsics"]["K"][0][0], 525.0)
            frames = manifest["frames"]
            self.assertEqual(frames[0]["frame_id"], 0)
            self.assertEqual(frames[0]["split"], "val")
            self.assertEqual(frames[1]["split"], "train")
            self.assertEqual(frames[0]["camera_center_world_m"], [0.0, 0.0, 0.0])
            self.assertEqual(frames[1]["camera_center_world_m"], [0.1, 0.0, 0.0])

    def test_prepare_manifest_block_split_uses_tail_validation_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rgbd_dataset_freiburg1_xyz"
            _write_minimal_tum_sequence(root, frame_count=10)
            output = Path(tmp) / "manifest_block.json"

            manifest = prepare_tum_rgbd_manifest(
                root,
                output,
                split_policy="block",
                val_fraction=0.2,
            )

            frames = manifest["frames"]
            train_ids = {frame["frame_id"] for frame in frames if frame["split"] == "train"}
            val_ids = {frame["frame_id"] for frame in frames if frame["split"] == "val"}
            self.assertEqual(val_ids, {8, 9})
            self.assertTrue(train_ids.isdisjoint(val_ids))
            self.assertEqual(manifest["split"]["policy"], "block")
            self.assertEqual(manifest["split"]["val_fraction"], 0.2)

    def test_safe_extract_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "unsafe.tar"
            with tarfile.open(archive, "w") as tar:
                payload = b"bad"
                info = tarfile.TarInfo("../bad.txt")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

            with self.assertRaisesRegex(ValueError, "parent traversal"):
                safe_extract_tar(archive, root / "out")


def _write_minimal_tum_sequence(root: Path, *, frame_count: int) -> None:
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)
    rgb_lines = ["# rgb"]
    depth_lines = ["# depth"]
    gt_lines = ["# groundtruth"]
    for index in range(frame_count):
        timestamp = 1.0 + index * 0.033
        rgb_name = f"rgb/{index:06d}.png"
        depth_name = f"depth/{index:06d}.png"
        (root / rgb_name).write_bytes(b"rgb")
        (root / depth_name).write_bytes(b"depth")
        rgb_lines.append(f"{timestamp:.6f} {rgb_name}")
        depth_lines.append(f"{timestamp + 0.001:.6f} {depth_name}")
        gt_lines.append(f"{timestamp + 0.002:.6f} {index * 0.1:.6f} 0 0 0 0 0 1")
    (root / "rgb.txt").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")
    (root / "depth.txt").write_text("\n".join(depth_lines) + "\n", encoding="utf-8")
    (root / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
