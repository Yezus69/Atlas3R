import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Phase6AStatusTest(unittest.TestCase):
    def test_next_task_handoff_points_to_phase6b_live_video_capture(self) -> None:
        next_task = (ROOT / "docs" / "status" / "next_task.md").read_text(encoding="utf-8")
        progress = (ROOT / "docs" / "status" / "progress.md").read_text(encoding="utf-8")

        self.assertIn("Phase 6B - Live/Video Input and Calibration Capture", next_task)
        self.assertIn("validated `atlas3r_recording` folders", next_task)
        self.assertIn("phase6a_product_slice_mapper_recording_mesh_report.md", progress)


if __name__ == "__main__":
    unittest.main()
