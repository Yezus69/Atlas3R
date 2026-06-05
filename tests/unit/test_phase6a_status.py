import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Phase6HStatusTest(unittest.TestCase):
    def test_next_task_handoff_points_to_core_phase_a3(self) -> None:
        next_task = (ROOT / "docs" / "status" / "next_task.md").read_text(encoding="utf-8")
        progress = (ROOT / "docs" / "status" / "progress.md").read_text(encoding="utf-8")

        self.assertIn(
            "Core Phase A3 - Replace Or Harden SMGT-Tiny Before Object Fusion",
            next_task,
        )
        self.assertIn("before object/dynamic fusion", next_task)
        self.assertIn("core_smgt_tiny_a2_generalization_report.md", progress)


if __name__ == "__main__":
    unittest.main()
