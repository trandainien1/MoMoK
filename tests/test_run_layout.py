import json
import tempfile
import unittest
from pathlib import Path


from ablations.logging_contract import RunLayout, record_failure


class RunLayoutTests(unittest.TestCase):
    def test_layout_is_namespaced_by_dataset_variant_seed_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = RunLayout.create(
                Path(tmp),
                dataset="MKG-W",
                variant="no_exid",
                seed=10010,
                config_hash="abc123",
            )

            self.assertEqual(
                layout.root,
                Path(tmp) / "MKG-W" / "no_exid" / "10010" / "abc123",
            )
            for directory in ("logs", "metrics", "checkpoints", "reports", "exports"):
                self.assertTrue((layout.root / directory).is_dir())

    def test_failure_record_contains_traceback_and_last_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = RunLayout.create(
                Path(tmp),
                dataset="MKG-W",
                variant="full",
                seed=10010,
                config_hash="abc123",
            )
            try:
                raise RuntimeError("synthetic smoke failure")
            except RuntimeError:
                record_failure(layout, phase="resume", last_completed_epoch=4)

            failure = json.loads((layout.reports / "failure.json").read_text())
            self.assertEqual(failure["phase"], "resume")
            self.assertEqual(failure["last_completed_epoch"], 4)
            self.assertIn("RuntimeError", failure["traceback"])
            self.assertEqual(failure["exit_status"], "failed")


if __name__ == "__main__":
    unittest.main()
