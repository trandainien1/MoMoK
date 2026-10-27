import json
import tempfile
import unittest
from pathlib import Path

from ablations.artifact_bundle import (
    create_resume_bundle,
    extract_verified_bundle,
    verify_resume_bundle,
)
from ablations.logging_contract import RunLayout


class ArtifactBundleTests(unittest.TestCase):
    def test_roundtrip_preserves_files_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = RunLayout.create(
                Path(tmp) / "runs",
                dataset="MKG-W",
                variant="no_gating_noise",
                seed=10010,
                config_hash="abc123",
            )
            (layout.root / "config.json").write_text(json.dumps({"dataset": "MKG-W"}))
            (layout.checkpoints / "checkpoint_latest.pt").write_bytes(b"checkpoint")
            (layout.logs / "session_001.log").write_text("RESUME_OK\n")

            bundle = create_resume_bundle(layout)
            manifest = verify_resume_bundle(bundle)

            self.assertIn("checkpoints/checkpoint_latest.pt", manifest["files"])
            restored = Path(tmp) / "restored"
            extract_verified_bundle(bundle, restored)
            self.assertEqual(
                (restored / "checkpoints/checkpoint_latest.pt").read_bytes(),
                b"checkpoint",
            )
            self.assertEqual(
                (restored / "logs/session_001.log").read_text(), "RESUME_OK\n"
            )

    def test_bundle_requires_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = RunLayout.create(
                Path(tmp) / "runs",
                dataset="MKG-W",
                variant="full",
                seed=10010,
                config_hash="abc123",
            )
            with self.assertRaises(FileNotFoundError):
                create_resume_bundle(layout)


if __name__ == "__main__":
    unittest.main()
