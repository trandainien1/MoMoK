from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]


class OptionalEstimatorCheckpointTests(unittest.TestCase):
    def test_checkpoint_schema_makes_estimator_state_optional(self):
        from ablations.checkpoint_contract import required_state_keys

        with_exid = required_state_keys(use_exid=True)
        without_exid = required_state_keys(use_exid=False)

        self.assertIn("estimator_state_dict", with_exid)
        self.assertIn("optimizer_mi_state_dict", with_exid)
        self.assertNotIn("estimator_state_dict", without_exid)
        self.assertNotIn("optimizer_mi_state_dict", without_exid)

    def test_checkpoint_metadata_contains_variant_identity(self):
        from ablations.checkpoint_contract import build_checkpoint_metadata

        metadata = build_checkpoint_metadata(
            variant_id="k_sweep",
            config_hash="abc",
            dataset="MKG-W",
            source_commit="commit",
            completed_epoch=3,
            n_exp=1,
        )

        self.assertEqual(metadata["variant_id"], "k_sweep")
        self.assertEqual(metadata["config_hash"], "abc")
        self.assertEqual(metadata["completed_epoch"], 3)
        self.assertEqual(metadata["n_exp"], 1)


if __name__ == "__main__":
    unittest.main()
