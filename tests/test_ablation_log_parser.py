import tempfile
import unittest
from pathlib import Path

from scripts.parse_momok_ablation_log import parse_raw_ablation_log


def make_log(*, include_terminal=True, eval_epochs=(100, 2000)):
    chunks = [
        "Namespace(lr=0.001, mu=0.0001, dim=200, r_dim=256, dataset='MKG-W', epochs=2000, batch_size=1024, n_exp=3, seed=10010, eval_freq=100, variant='no_gating_noise')\n"
    ]
    for epoch in eval_epochs:
        chunks.append(f"Epoch {epoch} , average loss 0.125 , epoch_time 12.5\n")
        chunks.append(f"Epoch: {epoch}\n")
        chunks.append(
            "test_Hits@100: 0.60 test_Hits@10: 0.45 test_Hits@3: 0.37 "
            "test_Hits@1: 0.29 test_MR: 1000.0 test_MRR: 0.35\n"
        )
    if include_terminal:
        chunks.append(
            "Test set results:\n"
            "test_Hits@100: 0.61 test_Hits@10: 0.4543 test_Hits@3: 0.3728 "
            "test_Hits@1: 0.2983 test_MR: 729.5 test_MRR: 0.3503\n"
        )
    return "".join(chunks)


class RawAblationLogParserTests(unittest.TestCase):
    def parse(self, content):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ablation_smoke.log"
            path.write_text(content)
            return parse_raw_ablation_log(path)

    def test_parses_observed_epoch_loss_eval_and_terminal_legacy_metrics(self):
        rows, summary, provenance = self.parse(make_log())
        self.assertEqual([row["epoch"] for row in rows], [100, 2000])
        self.assertEqual(rows[0]["train_main_loss"], 0.125)
        self.assertEqual(rows[0]["epoch_time_sec"], 12.5)
        self.assertEqual(rows[0]["eval_MRR"], 0.35)
        self.assertEqual(rows[0]["eval_Hits@1"], 0.29)
        self.assertNotIn("learning_rate", rows[0])
        self.assertEqual(summary["legacy_final_metrics"]["MRR"], 0.3503)
        self.assertEqual(summary["epoch_2000_metrics"]["MRR"], 0.35)
        self.assertNotEqual(
            summary["epoch_2000_metrics"]["MRR"],
            summary["legacy_final_metrics"]["MRR"],
        )
        self.assertEqual(provenance["tracking_mode"], "retrospective_import_from_kaggle_logs")
        self.assertEqual(summary["fabricated_points"], 0)

    def test_missing_terminal_epoch_and_missing_final_summary_fail_closed(self):
        with self.assertRaises(ValueError):
            self.parse(make_log(eval_epochs=(100,), include_terminal=True))
        content = make_log(include_terminal=False)
        content = content.replace("Epoch 200 , average loss", "Epoch 2000 , average loss")
        content = content.replace("Epoch: 200\n", "Epoch: 2000\n")
        with self.assertRaises(ValueError):
            self.parse(content)

    def test_repeated_epoch_eval_uses_last_record_without_adding_a_point(self):
        content = make_log()
        repeated = "Epoch: 2000\ntest_Hits@10: 0.46 test_Hits@3: 0.38 test_Hits@1: 0.30 test_MR: 900 test_MRR: 0.36\n"
        content += repeated
        rows, summary, _ = self.parse(content)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["eval_MRR"], 0.36)
        self.assertEqual(summary["evaluation_points"], 2)


if __name__ == "__main__":
    unittest.main()
