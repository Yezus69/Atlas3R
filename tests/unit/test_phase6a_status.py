import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Phase6HStatusTest(unittest.TestCase):
    def test_next_task_handoff_points_to_core_phase_a4(self) -> None:
        next_task = (ROOT / "docs" / "status" / "next_task.md").read_text(encoding="utf-8")
        progress = (ROOT / "docs" / "status" / "progress.md").read_text(encoding="utf-8")

        self.assertIn(
            "Core Phase A4 - Broaden Student Validation And First-Class Profiling",
            next_task,
        )
        self.assertIn("before object/dynamic fusion", next_task)
        self.assertIn("core_smgt_small_v2_measured_pseudo_report.md", progress)


if __name__ == "__main__":
    unittest.main()
