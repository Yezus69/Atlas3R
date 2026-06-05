from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from atlas3r.offline.object_ledger import write_object_ledger
from atlas3r.offline.proposal_cache import ProposalCacheResult
from atlas3r.offline.run_manifest import ensure_run_tree


class ObjectLedgerTest(unittest.TestCase):
    def test_object_ledger_exists_without_inventing_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = ensure_run_tree(Path(tmp) / "run")
            failures = []
            proposals = ProposalCacheResult(
                status="partial",
                manifest_path="proposals/proposal_manifest.json",
                streams=({"teacher_name": "sam_dino", "status": "unavailable"},),
                debug_depth_records=(),
                depth_proposal_available=False,
                debug_geometry_mode="none",
            )

            result = write_object_ledger(run_dir, proposal_cache=proposals, failure_points=failures)

            payload = json.loads((run_dir / result.object_ledger_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "unavailable")
            self.assertEqual(payload["objects"], [])
            self.assertFalse(payload["invented_objects"])


if __name__ == "__main__":
    unittest.main()
