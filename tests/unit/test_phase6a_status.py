import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Phase6HStatusTest(unittest.TestCase):
    def test_next_task_handoff_points_to_phase6i_student_training(self) -> None:
        next_task = (ROOT / "docs" / "status" / "next_task.md").read_text(encoding="utf-8")
        progress = (ROOT / "docs" / "status" / "progress.md").read_text(encoding="utf-8")

        self.assertIn(
            "Phase 6I - Train Tiny Student From Measured And Teacher Temporal Caches",
            next_task,
        )
        self.assertIn("TeacherTemporalCacheDataset", next_task)
        self.assertIn("phase6h_teacher_stitch_cache_report.md", progress)


if __name__ == "__main__":
    unittest.main()
