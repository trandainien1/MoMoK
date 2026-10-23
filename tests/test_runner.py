import unittest


from ablations.cli import parse_args
from ablations.runner import build_train_command


class RunnerTests(unittest.TestCase):
    def test_command_contains_variant_and_smoke_boundaries(self):
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
                "/tmp/k1",
            ]
        )
        command = build_train_command(args, train_script="train.py")

        self.assertIn("--variant", command)
        self.assertIn("k_sweep", command)
        self.assertIn("--n_exp", command)
        self.assertIn("1", command)
        self.assertIn("--max_train_batches", command)
        self.assertIn("2", command)
        self.assertIn("--run_mode", command)
        self.assertIn("smoke", command)


if __name__ == "__main__":
    unittest.main()
