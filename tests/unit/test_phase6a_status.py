import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Phase6HStatusTest(unittest.TestCase):
    def test_next_task_handoff_points_to_core_phase_b(self) -> None:
        next_task = (ROOT / "docs" / "status" / "next_task.md").read_text(encoding="utf-8")
        progress = (ROOT / "docs" / "status" / "progress.md").read_text(encoding="utf-8")

        self.assertIn(
            "Core Phase B - Object/Dynamic Teacher Labels And Object-Aware Sparse Fusion",
            next_task,
        )
        self.assertIn("object/dynamic", next_task)
        self.assertIn("core_smgt_tiny_student_map_report.md", progress)


if __name__ == "__main__":
    unittest.main()
