import hashlib
import tempfile
import unittest
from pathlib import Path


from ablations.checkpoints import atomic_save, validate_payload


class CheckpointAdapterTests(unittest.TestCase):
    def test_atomic_save_writes_checksum_and_replaces_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoints" / "checkpoint_latest.bin"
            digest = atomic_save(b"checkpoint", path)

            self.assertEqual(digest, hashlib.sha256(b"checkpoint").hexdigest())
            self.assertEqual(path.read_bytes(), b"checkpoint")
            self.assertEqual(path.with_suffix(".bin.sha256").read_text().split()[0], digest)
            self.assertFalse(path.with_suffix(".bin.tmp").exists())

    def test_validate_payload_rejects_wrong_config_and_missing_estimator_state(self):
        payload = {
            "format_version": 1,
            "completed_epoch": 2,
            "variant_id": "no_exid",
            "config_hash": "abc",
            "dataset": "MKG-W",
            "source_commit": "commit",
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "lr_scheduler_state_dict": {},
            "best_validation_metrics": {},
            "python_random_state": None,
            "numpy_rng_state": None,
            "torch_rng_state": None,
            "torch_cuda_rng_state_all": [],
            "corpus_train_indices": [],
            "config": {"dataset": "MKG-W"},
        }

        self.assertEqual(
            validate_payload(payload, expected_config={"dataset": "MKG-W"}, use_exid=False),
            payload,
        )
        with self.assertRaises(ValueError):
            validate_payload(payload, expected_config={"dataset": "MKG-Y"}, use_exid=False)
        with self.assertRaises(ValueError):
            validate_payload(payload, expected_config={"dataset": "MKG-W"}, use_exid=True)


if __name__ == "__main__":
    unittest.main()
