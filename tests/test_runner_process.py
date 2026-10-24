import json
import sys
import tempfile
import unittest
from pathlib import Path


from ablations.cli import parse_args
from ablations.runner import run_process
from ablations.variants import config_fingerprint


class RunnerProcessTests(unittest.TestCase):
    def test_process_wrapper_captures_session_log_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(
                [
                    "--dataset",
                    "MKG-W",
                    "--variant",
                    "k_sweep",
                    "--n_exp",
                    "1",
                    "--lr",
                    "0.001",
                    "--dim",
                    "200",
                    "--epochs",
                    "2",
                    "--max_train_batches",
                    "2",
                    "--run_mode",
                    "smoke",
                    "--output_dir",
                    tmp,
                ]
            )
            script = Path(tmp) / "stub.py"
            script.write_text("print('SMOKE_STUB=PASS')\n")
            result = run_process(args, train_script=script, base_output=Path(tmp) / "runs")

            self.assertEqual(result.returncode, 0)
            self.assertIn("SMOKE_STUB=PASS", result.stdout)
            self.assertTrue(result.layout.logs.joinpath("session_001.log").is_file())
            manifest = json.loads((result.layout.root / "config.json").read_text())
            self.assertEqual(manifest["variant"], "k_sweep")
            self.assertEqual(manifest["config_hash"], config_fingerprint(manifest["config"]))

    def test_process_isolates_output_and_exports_resume_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(
                [
                    "--dataset", "MKG-W", "--variant", "k_sweep", "--n_exp", "1",
                    "--lr", "0.001", "--dim", "200", "--epochs", "2",
                    "--max_train_batches", "2", "--run_mode", "smoke",
                    "--output_dir", tmp,
                ]
            )
            script = Path(tmp) / "stub.py"
            script.write_text(
                "import argparse, pathlib\n"
                "p=argparse.ArgumentParser(); p.add_argument('--checkpoint_dir'); p.add_argument('--output_dir'); a,_=p.parse_known_args()\n"
                "pathlib.Path(a.checkpoint_dir).mkdir(parents=True, exist_ok=True)\n"
                "pathlib.Path(a.checkpoint_dir, 'checkpoint_latest.pt').write_bytes(b'checkpoint')\n"
            )
            result = run_process(args, train_script=script, base_output=Path(tmp) / "runs")

            self.assertEqual(result.returncode, 0)
            self.assertTrue((result.layout.exports / "resume_bundle.zip").is_file())
            self.assertFalse((Path(tmp) / "checkpoint_latest.pt").exists())


if __name__ == "__main__":
    unittest.main()
