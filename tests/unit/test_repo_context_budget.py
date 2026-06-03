import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

LINE_BUDGETS = {
    "AGENTS.md": 140,
    "PLANS.md": 220,
    "docs/08_API_CONTRACTS.md": 450,
    "docs/status/active_task.md": 80,
    "docs/status/progress.md": 180,
    "docs/status/decisions.md": 180,
    "docs/status/next_task.md": 150,
}


class RepoContextBudgetTest(unittest.TestCase):
    def test_markdown_files_stay_within_context_budget(self) -> None:
        for relative_path, max_lines in LINE_BUDGETS.items():
            path = REPO_ROOT / relative_path
            with self.subTest(path=relative_path):
                line_count = len(path.read_text(encoding="utf-8").splitlines())
                self.assertLessEqual(
                    line_count,
                    max_lines,
                    f"{relative_path} has {line_count} lines; budget is {max_lines}",
                )

    def test_next_task_points_to_phase_5f(self) -> None:
        next_task = (REPO_ROOT / "docs/status/next_task.md").read_text(encoding="utf-8")
        self.assertIn("Phase 5F", next_task)
        self.assertIn("external teacher data", next_task)


if __name__ == "__main__":
    unittest.main()
