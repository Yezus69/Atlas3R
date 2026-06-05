import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Phase6GStatusTest(unittest.TestCase):
    def test_next_task_handoff_points_to_phase6h_distillation(self) -> None:
        next_task = (ROOT / "docs" / "status" / "next_task.md").read_text(encoding="utf-8")
        progress = (ROOT / "docs" / "status" / "progress.md").read_text(encoding="utf-8")

        self.assertIn(
            "Phase 6H - Distill RGB Teacher Geometry Into Student Training Data", next_task
        )
        self.assertIn("runtime map-rgb-teacher", next_task)
        self.assertIn("phase6g_rgb_teacher_map_report.md", progress)


if __name__ == "__main__":
    unittest.main()
