import unittest


from ablations.runtime_contract import (
    estimator_should_run,
    inference_heads,
    training_heads,
)
from ablations.variants import resolve_variant


class RuntimeContractTests(unittest.TestCase):
    def test_full_uses_all_heads_and_estimator(self):
        config = resolve_variant("full", n_exp=3, mu=0.0001)

        self.assertEqual(training_heads(config), ("structure", "image", "text", "joint"))
        self.assertEqual(inference_heads(config), ("structure", "image", "text", "joint"))
        self.assertTrue(estimator_should_run(config))

    def test_single_modality_variant_has_one_training_and_inference_head(self):
        for variant_id, head in (
            ("structure_only", "structure"),
            ("image_only", "image"),
            ("text_only", "text"),
            ("joint_only", "joint"),
        ):
            config = resolve_variant(variant_id, n_exp=3, mu=0.0001)
            self.assertEqual(training_heads(config), (head,))
            self.assertEqual(inference_heads(config), (head,))

    def test_joint_loss_only_keeps_joint_inference(self):
        config = resolve_variant("no_joint_training_loss_only", n_exp=3, mu=0.0001)

        self.assertEqual(training_heads(config), ("structure", "image", "text"))
        self.assertEqual(inference_heads(config), ("structure", "image", "text", "joint"))

    def test_remove_joint_head_excludes_joint_everywhere(self):
        config = resolve_variant("no_joint_head", n_exp=3, mu=0.0001)

        self.assertEqual(training_heads(config), ("structure", "image", "text"))
        self.assertEqual(inference_heads(config), ("structure", "image", "text"))

    def test_mu_zero_proxy_runs_estimator_but_no_exid_does_not(self):
        proxy = resolve_variant("mu_zero_proxy", n_exp=3, mu=0.0)
        no_exid = resolve_variant("no_exid", n_exp=3, mu=0.0)

        self.assertTrue(estimator_should_run(proxy))
        self.assertFalse(estimator_should_run(no_exid))

    def test_k_one_sweep_does_not_run_estimator(self):
        config = resolve_variant("k_sweep", n_exp=1, mu=0.0001)

        self.assertEqual(training_heads(config), ("structure", "image", "text", "joint"))
        self.assertFalse(estimator_should_run(config))


if __name__ == "__main__":
    unittest.main()
