import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_ablation_kaggle_notebook import build_notebook


class AblationNotebookBuilderTests(unittest.TestCase):
    def test_smoke_failure_emits_child_log_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "smoke.ipynb"
            build_notebook(
                output=output,
                dataset="MKG-W",
                variant="no_gating_noise",
                lr=0.001,
                mu=0.0001,
                dim=200,
                n_exp=3,
                epochs=2,
                run_mode="smoke",
            )
            notebook = json.loads(output.read_text())
            run_source = "".join(notebook["cells"][7]["source"])
            self.assertIn("TRAIN_LOG_BEGIN", run_source)
            self.assertIn("TRAIN_LOG_END", run_source)


if __name__ == "__main__":
    unittest.main()
