import dataclasses
import unittest

from ablations.wandb_publication import (
    BASELINE_METRICS,
    BASELINE_RUN_ID,
    PUBLICATION_VARIANTS,
    VariantPublicationError,
    comparison_rows,
    publication_for,
    validate_dataset_manifest,
)


class PublicationRegistryTests(unittest.TestCase):
    def test_registry_contains_only_three_completed_variants(self):
        self.assertEqual(
            set(PUBLICATION_VARIANTS),
            {"no_gating_noise", "no_exid", "no_relation_temperature"},
        )
        self.assertEqual(len({item.run_id for item in PUBLICATION_VARIANTS.values()}), 3)

    def test_records_are_frozen_and_pin_source_and_baseline(self):
        item = publication_for("no_exid")
        self.assertTrue(dataclasses.is_dataclass(item))
        self.assertTrue(item.__dataclass_params__.frozen)
        self.assertEqual(item.source_commit, "99c2df114d48c79708ea0608644b685183d81bd9")
        self.assertEqual(item.baseline_run_id, BASELINE_RUN_ID)
        self.assertEqual(item.baseline_label, "Full Model (Proposed Method)")

    def test_unknown_variant_fails_closed(self):
        with self.assertRaises(VariantPublicationError):
            publication_for("full")

    def test_dataset_manifest_requires_mkgw_fingerprint_and_feature_shapes(self):
        item = publication_for("no_gating_noise")
        manifest = {
            "entities": 15000,
            "relations": 169,
            "train": 34196,
            "valid": 4276,
            "test": 4274,
            "image_shape": [15000, 383],
            "text_shape": [15000, 384],
        }
        validate_dataset_manifest(manifest, item)
        manifest["relations"] = 28
        with self.assertRaises(VariantPublicationError):
            validate_dataset_manifest(manifest, item)

    def test_no_exid_contract_is_explicit(self):
        item = publication_for("no_exid")
        self.assertEqual(item.config["mu"], 0.0)
        self.assertIs(item.config["use_exid"], False)
        self.assertEqual(item.config["exid_mode"], "disabled")

    def test_exact_legacy_final_metrics_are_registered(self):
        expected = {
            "no_gating_noise": {
                "MRR": 0.3503,
                "Hits@1": 0.2983,
                "Hits@3": 0.3728,
                "Hits@10": 0.4543,
            },
            "no_exid": {
                "MRR": 0.3509,
                "Hits@1": 0.2968,
                "Hits@3": 0.3724,
                "Hits@10": 0.4586,
            },
            "no_relation_temperature": {
                "MRR": 0.3586,
                "Hits@1": 0.3059,
                "Hits@3": 0.3759,
                "Hits@10": 0.4628,
            },
        }
        for variant_id, metrics in expected.items():
            self.assertEqual(publication_for(variant_id).final_metrics, metrics)
        self.assertEqual(BASELINE_METRICS["MRR"], 0.3514488296)

    def test_comparison_rows_include_proposed_method_baseline_and_deltas(self):
        rows = comparison_rows(list(PUBLICATION_VARIANTS.values()))
        self.assertEqual(rows[0][0], "Full Model (Proposed Method)")
        self.assertEqual(rows[0][1], BASELINE_RUN_ID)
        variant_ids = {row[0] for row in rows if row[0] != "Full Model (Proposed Method)"}
        self.assertEqual(variant_ids, set(PUBLICATION_VARIANTS))
        self.assertTrue(all(len(row) == 8 for row in rows))


if __name__ == "__main__":
    unittest.main()
