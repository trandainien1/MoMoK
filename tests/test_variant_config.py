import unittest


from ablations.variants import VariantConfigError, config_fingerprint, resolve_variant


class VariantConfigTests(unittest.TestCase):
    def test_full_preserves_baseline_defaults(self):
        config = resolve_variant("full", n_exp=3, mu=0.0001)

        self.assertEqual(config.variant_id, "full")
        self.assertEqual(config.active_heads, ("structure", "image", "text", "joint"))
        self.assertTrue(config.use_relation_temperature)
        self.assertTrue(config.use_gating_noise)
        self.assertTrue(config.use_adaptive_fusion)
        self.assertTrue(config.train_joint_head)
        self.assertTrue(config.infer_joint_head)
        self.assertTrue(config.use_exid)
        self.assertEqual(config.exid_mode, "active")
        self.assertEqual(config.n_exp, 3)
        self.assertEqual(config.mu, 0.0001)

    def test_k_one_disables_exid_only_in_explicit_k_sweep_variant(self):
        config = resolve_variant("k_sweep", n_exp=1, mu=0.0001)

        self.assertEqual(config.variant_id, "k_sweep")
        self.assertEqual(config.n_exp, 1)
        self.assertFalse(config.use_exid)
        self.assertEqual(config.exid_mode, "disabled_k1")
        self.assertEqual(config.mu, 0.0)

    def test_k_one_is_rejected_for_full_model(self):
        with self.assertRaises(VariantConfigError):
            resolve_variant("full", n_exp=1, mu=0.0001)

    def test_mu_zero_proxy_keeps_estimator_overhead_explicit(self):
        config = resolve_variant("mu_zero_proxy", n_exp=3, mu=0.0)

        self.assertTrue(config.use_exid)
        self.assertEqual(config.exid_mode, "mu_zero_proxy")
        self.assertEqual(config.mu, 0.0)

    def test_no_exid_disables_estimator_and_requires_zero_mu(self):
        config = resolve_variant("no_exid", n_exp=3, mu=0.0)

        self.assertFalse(config.use_exid)
        self.assertEqual(config.exid_mode, "disabled")

        with self.assertRaises(VariantConfigError):
            resolve_variant("no_exid", n_exp=3, mu=0.0001)

    def test_variant_ids_are_explicit_and_unknown_ids_fail(self):
        for variant_id in (
            "structure_only",
            "image_only",
            "text_only",
            "joint_only",
            "no_relation_temperature",
            "no_gating_noise",
            "no_adaptive_fusion",
            "no_joint_training_loss_only",
            "no_joint_head",
            "mu_sweep",
        ):
            config = resolve_variant(variant_id, n_exp=3, mu=0.0001)
            self.assertEqual(config.variant_id, variant_id)

        with self.assertRaises(VariantConfigError):
            resolve_variant("not_a_variant", n_exp=3, mu=0.0001)

    def test_config_fingerprint_is_deterministic_and_order_independent(self):
        first = config_fingerprint({"dataset": "MKG-W", "seed": 10010, "variant": "full"})
        second = config_fingerprint({"variant": "full", "seed": 10010, "dataset": "MKG-W"})

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
