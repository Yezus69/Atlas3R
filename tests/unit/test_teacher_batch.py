import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from atlas3r.api import FramePacket
from atlas3r.data import (
    NPZFrameSource,
    teacher_frame_batch_from_frame_packets,
    teacher_frame_batch_from_rgb_source,
    write_frame_source_smoke_fixture,
    write_synthetic_cube_room_session,
)
from atlas3r.models.adapters import FixtureCubeRoomTeacherAdapter, FrameBatch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _K() -> np.ndarray:
    return np.array(
        [
            [80.0, 0.0, 1.5],
            [0.0, 80.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _frame_packet(frame_id: int) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp_ns=frame_id * 1_000_000,
        rgb_u8=np.zeros((2, 3, 3), dtype=np.uint8),
        rgb_model=np.zeros((3, 2, 3), dtype=np.float32),
        K_original=None,
        K_model=_K(),
        distortion=None,
        resize_transform=np.eye(3, dtype=np.float32),
        camera_metadata={"source_format": "unit_test"},
    )


class TeacherFrameBatchBridgeTest(unittest.TestCase):
    def test_smoke_frame_packets_convert_to_teacher_frame_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packets = write_frame_source_smoke_fixture(tmp)

        batch = teacher_frame_batch_from_frame_packets(
            packets,
            batch_id="unit-teacher-batch",
            metadata={"session_path": "fixture.atlas3r"},
        )

        self.assertIsInstance(batch, FrameBatch)
        self.assertEqual(batch.batch_id, "unit-teacher-batch")
        self.assertEqual(tuple(packet.frame_id for packet in batch.frames), (0, 1))
        self.assertIs(batch.frames[0], packets[0])
        self.assertEqual(batch.metadata["source"], "frame_packets")
        self.assertEqual(batch.metadata["frame_count"], 2)
        self.assertEqual(batch.metadata["frame_ids"], (0, 1))
        self.assertIs(batch.metadata["input_order_preserved"], True)
        self.assertEqual(batch.metadata["source_formats"], ("npz",))
        self.assertEqual(batch.metadata["session_path"], "fixture.atlas3r")

    def test_frame_id_order_is_preserved_without_sorting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packets = write_frame_source_smoke_fixture(tmp)

        batch = teacher_frame_batch_from_frame_packets((packets[1], packets[0]))

        self.assertEqual(tuple(packet.frame_id for packet in batch.frames), (1, 0))
        self.assertEqual(batch.metadata["frame_ids"], (1, 0))

    def test_rgb_frame_source_wrapper_builds_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_frame_source_smoke_fixture(tmp)
            source = NPZFrameSource(Path(tmp) / "frame_source_smoke.npz", frame_id_start=4)
            batch = teacher_frame_batch_from_rgb_source(source, batch_id="source-batch")

        self.assertEqual(batch.batch_id, "source-batch")
        self.assertEqual(tuple(packet.frame_id for packet in batch.frames), (4, 5))
        self.assertEqual(batch.metadata["source"], "rgb_frame_source")
        self.assertEqual(batch.metadata["frame_ids"], (4, 5))

    def test_duplicate_frame_ids_are_rejected(self) -> None:
        first = _frame_packet(7)
        duplicate = replace(_frame_packet(8), frame_id=7)

        with self.assertRaisesRegex(ValueError, r"frames\[1\]\.frame_id.*duplicate"):
            teacher_frame_batch_from_frame_packets((first, duplicate))

    def test_empty_sequences_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "frames.*at least one"):
            teacher_frame_batch_from_frame_packets(())

    def test_non_frame_packet_items_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"frames\[1\].*FramePacket"):
            teacher_frame_batch_from_frame_packets((_frame_packet(0), object()))

    def test_reserved_metadata_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"metadata\.frame_ids.*reserved"):
            teacher_frame_batch_from_frame_packets(
                (_frame_packet(0),),
                metadata={"frame_ids": (99,)},
            )

    def test_fixture_teacher_adapter_accepts_bridge_batch_without_external_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_path = Path(tmp) / "fixture.atlas3r"
            write_synthetic_cube_room_session(session_path)
            packets = tuple(_frame_packet(frame_id) for frame_id in (0, 1, 2))

            batch = teacher_frame_batch_from_frame_packets(
                packets,
                metadata={"session_path": str(session_path)},
            )
            prediction = FixtureCubeRoomTeacherAdapter().predict(batch)

        self.assertEqual(prediction.adapter_name, "fixture-cube-room")
        self.assertEqual(
            tuple(
                frame_prediction.pose.frame_id for frame_prediction in prediction.frame_predictions
            ),
            (0, 1, 2),
        )

    def test_teacher_batch_imports_do_not_load_heavy_dependencies(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        code = (
            "import sys\n"
            "import atlas3r.data.teacher_batch\n"
            "import atlas3r.data as data\n"
            "getattr(data, 'teacher_frame_batch_from_frame_packets')\n"
            "getattr(data, 'teacher_frame_batch_from_rgb_source')\n"
            "heavy = {'torch', 'tensorflow', 'jax', 'cv2', 'av', 'imageio', 'PIL'} & "
            "set(sys.modules)\n"
            "unexpected = heavy | ({'atlas3r.mapping'} & set(sys.modules)) | "
            "({'atlas3r.runtime'} & set(sys.modules))\n"
            "message = 'unexpected imports: ' + ', '.join(sorted(unexpected))\n"
            "raise SystemExit(message if unexpected else 0)\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
