from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]


class TorchCheckpointWiringTests(unittest.TestCase):
    def test_checkpoint_adapter_contains_full_state_hooks(self):
        source = (ROOT / "ablations" / "torch_checkpoint.py").read_text()

        for marker in (
            "model_state_dict",
            "optimizer_state_dict",
            "lr_scheduler_state_dict",
            "python_random_state",
            "numpy_rng_state",
            "torch_rng_state",
            "corpus_train_indices",
            "atomic_save",
            "restore_checkpoint",
        ):
            self.assertIn(marker, source)

    def test_train_source_wires_resume_and_checkpoint_interval(self):
        source = (ROOT / "train.py").read_text()

        self.assertIn("save_checkpoint", source)
        self.assertIn("restore_checkpoint", source)
        self.assertIn("checkpoint_dir", source)
        self.assertIn("completed_epoch", source)


if __name__ == "__main__":
    unittest.main()
