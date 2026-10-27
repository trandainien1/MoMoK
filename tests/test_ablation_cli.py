import unittest


from ablations.cli import parse_args


class AblationCliTests(unittest.TestCase):
    def test_cli_resolves_variant_and_runtime_controls(self):
        args = parse_args(
            [
                "--dataset",
                "MKG-W",
                "--variant",
                "no_gating_noise",
                "--n_exp",
                "3",
                "--mu",
                "0.0001",
                "--lr",
                "0.001",
                "--dim",
                "200",
                "--run_mode",
                "smoke",
                "--max_train_batches",
                "2",
                "--output_dir",
                "/tmp/momok-smoke",
            ]
        )

        self.assertEqual(args.dataset, "MKG-W")
        self.assertEqual(args.variant, "no_gating_noise")
        self.assertEqual(args.variant_config.variant_id, "no_gating_noise")
        self.assertEqual(args.max_train_batches, 2)
        self.assertEqual(args.run_mode, "smoke")

    def test_cli_rejects_invalid_k_one_full_configuration(self):
        with self.assertRaises(ValueError):
            parse_args(
                [
                    "--dataset",
                    "MKG-W",
                    "--variant",
                    "full",
                    "--n_exp",
                    "1",
                    "--lr",
                    "0.001",
                    "--dim",
                    "200",
                    "--output_dir",
                    "/tmp/full",
                ]
            )

    def test_cli_allows_explicit_k_one_sweep(self):
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
                "--output_dir",
                "/tmp/k1",
            ]
        )

        self.assertFalse(args.variant_config.use_exid)
        self.assertEqual(args.variant_config.exid_mode, "disabled_k1")


if __name__ == "__main__":
    unittest.main()
