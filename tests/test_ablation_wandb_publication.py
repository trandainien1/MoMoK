import dataclasses
import unittest

from ablations.wandb_publication import (
    BASELINE_METRICS,
    BASELINE_RUN_ID,
    PUBLICATION_VARIANTS,
    SAFE_OUTPUT_FILES,
    VariantPublicationError,
    comparison_rows,
    publication_for,
    safe_output_files,
    validate_dataset_manifest,
    validate_safe_output_dir,
    build_environment_rows,
    build_history_records,
    build_variant_summary,
    comparison_table_rows,
    existing_run_action,
)
import json
import csv
import tempfile
from pathlib import Path

from scripts.publish_momok_mkgw_ablation_wandb import publish_variant


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

    def test_download_allowlist_is_exact_and_checkpoint_binary_is_not_allowed(self):
        self.assertEqual(
            SAFE_OUTPUT_FILES,
            ("ablation_smoke.log", "dataset_manifest.json", "output_inventory.json", "checkpoint_final.pt.sha256"),
        )
        self.assertEqual(safe_output_files(), SAFE_OUTPUT_FILES)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "ablation_smoke.log").write_text("completed\n")
            (root / "dataset_manifest.json").write_text(json.dumps({
                "entities": 15000, "relations": 169, "train": 34196, "valid": 4276,
                "test": 4274, "image_shape": [15000, 383], "text_shape": [15000, 384],
            }))
            (root / "output_inventory.json").write_text(json.dumps(["checkpoints/checkpoint_final.pt"]))
            item = publication_for("no_gating_noise")
            (root / "checkpoint_final.pt.sha256").write_text(item.checkpoint_sha256 + "  checkpoint_final.pt\n")
            validate_safe_output_dir(root, item)
            (root / "checkpoint_final.pt").write_bytes(b"must never be downloaded")
            with self.assertRaises(VariantPublicationError):
                validate_safe_output_dir(root, item)

    def test_history_records_use_epoch_and_only_present_metrics(self):
        records = build_history_records([
            {"epoch": "100", "train_main_loss": "0.2", "eval_MRR": "0.31"},
            {"epoch": "200", "train_main_loss": "0.1", "mi_estimator_loss": "8.5"},
        ])
        self.assertEqual(records[0], {"epoch": 100, "train/loss": 0.2, "eval/MRR": 0.31})
        self.assertEqual(records[1], {"epoch": 200, "train/loss": 0.1, "train/mi_estimator_loss": 8.5})

    def test_comparison_table_contains_full_model_anchor_and_numeric_deltas(self):
        item = publication_for("no_gating_noise")
        rows = comparison_table_rows(item)
        self.assertEqual(rows[0][0], "MRR")
        self.assertEqual(rows[0][1], BASELINE_METRICS["MRR"])
        self.assertEqual(rows[0][2], 0.3503)
        self.assertAlmostEqual(rows[0][3], 0.3503 - BASELINE_METRICS["MRR"])
        self.assertTrue(all(isinstance(row[0], str) for row in rows))
        self.assertTrue(all(type(value) is float for row in rows for value in row[1:]))

    def test_environment_table_values_are_always_strings(self):
        item = publication_for("no_exid")
        rows = build_environment_rows(item, {"gpu": "Tesla T4", "python": "3.12"})
        self.assertTrue(rows)
        self.assertTrue(all(type(key) is str and type(value) is str for key, value in rows))

    def test_final_summary_uses_legacy_metrics_and_hash(self):
        item = publication_for("no_relation_temperature")
        summary = build_variant_summary(item, item.checkpoint_sha256)
        self.assertEqual(summary["completed_epoch"], 2000)
        self.assertEqual(summary["metric_policy"], "legacy_best_per_metric")
        self.assertEqual(summary["reproduced/MRR"], 0.3586)
        self.assertEqual(summary["checkpoint_sha256"], item.checkpoint_sha256)

    def test_existing_run_idempotency_skips_only_exactly_completed_publication(self):
        self.assertEqual(existing_run_action({}, [], "abc"), "resume_empty")
        self.assertEqual(
            existing_run_action({"publication_complete": True, "publication_fingerprint": "abc"}, [], "abc"),
            "skip_complete",
        )
        with self.assertRaises(VariantPublicationError):
            existing_run_action({"publication_tables_written": True}, [], "abc")
        with self.assertRaises(VariantPublicationError):
            existing_run_action({}, [{"epoch": 100}], "abc")

    def test_fake_adapter_publishes_one_run_history_tables_and_safe_artifact(self):
        class FakeTable:
            def __init__(self, columns, data):
                self.columns, self.data = columns, data

        class FakeArtifact:
            def __init__(self, name, type, metadata):
                self.name, self.type, self.metadata, self.files = name, type, metadata, []

            def add_file(self, path, name):
                self.files.append((Path(path), name))

        class FakeRun:
            def __init__(self, run_id):
                self.id, self.url, self.summary, self.config = run_id, "https://wandb.ai/test/run", {}, {}
                self.records, self.artifacts, self.metrics = [], [], []

            def define_metric(self, *args, **kwargs):
                self.metrics.append((args, kwargs))

            def log(self, row):
                self.records.append(row)

            def log_artifact(self, artifact):
                self.artifacts.append(artifact)

            def finish(self, **kwargs):
                pass

        class FakeApi:
            def run(self, path):
                raise RuntimeError("404 not found")

        class FakeWandb:
            Table = FakeTable
            Artifact = FakeArtifact
            api = FakeApi()
            run = None
            init_kwargs = None

            @classmethod
            def Api(cls):
                return cls.api

            @classmethod
            def init(cls, **kwargs):
                cls.init_kwargs = kwargs
                cls.run = FakeRun(kwargs["id"])
                cls.run.config = kwargs["config"]
                return cls.run

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            export = root / "export"
            evidence = root / "evidence"
            export.mkdir(); evidence.mkdir()
            rows = []
            for epoch in range(100, 2001, 100):
                rows.append({"epoch": epoch, "train_main_loss": 0.1,
                             "mi_estimator_loss": 2.0, "eval_MRR": 0.34,
                             "eval_Hits@1": 0.29, "eval_Hits@3": 0.36,
                             "eval_Hits@10": 0.44})
            with (export / "history.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            item = publication_for("no_gating_noise")
            (export / "history_summary.json").write_text(json.dumps({
                "fabricated_points": 0, "evaluation_points": 20,
                "training_points": 20, "first_epoch_found": 100,
                "last_epoch_found": 2000,
                "metric_policy": "legacy_best_per_metric",
                "legacy_final_metrics": item.final_metrics,
            }))
            (export / "audited_final_metrics.json").write_text(json.dumps(item.final_metrics))
            (export / "provenance.json").write_text(json.dumps({"tracking_mode": "retrospective_import_from_kaggle_logs"}))
            (evidence / "ablation_smoke.log").write_text("safe training log\n")
            (evidence / "dataset_manifest.json").write_text(json.dumps({
                "entities": 15000, "relations": 169, "train": 34196, "valid": 4276,
                "test": 4274, "image_shape": [15000, 383], "text_shape": [15000, 384],
            }))
            (evidence / "output_inventory.json").write_text(json.dumps(["checkpoints/checkpoint_final.pt"]))
            (evidence / "checkpoint_final.pt.sha256").write_text(item.checkpoint_sha256 + "  checkpoint_final.pt\n")

            result = publish_variant(item.variant_id, export, evidence, FakeWandb)

        self.assertEqual(result["run_id"], "mkgwngn1")
        self.assertEqual(FakeWandb.init_kwargs["resume"], "allow")
        self.assertEqual(FakeWandb.init_kwargs["id"], "mkgwngn1")
        self.assertEqual(len(FakeWandb.run.records), 22)
        self.assertEqual(FakeWandb.run.records[0]["epoch"], 100)
        self.assertEqual(FakeWandb.run.records[19]["epoch"], 2000)
        self.assertTrue(FakeWandb.run.summary["publication_complete"])
        artifact = FakeWandb.run.artifacts[0]
        self.assertEqual(len(artifact.files), 8)
        self.assertFalse(any(path.suffix == ".pt" for path, _ in artifact.files))


if __name__ == "__main__":
    unittest.main()
