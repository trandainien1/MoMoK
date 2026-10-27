import json
import re
import unittest
from pathlib import Path

import wandb_workspaces.reports.v2 as wr
from scripts.update_momok_main_wandb_report import (
    ABLATION_RUN_IDS,
    ablation_blocks,
    ablation_runset,
    blocks,
)


class AblationReportSpecTests(unittest.TestCase):
    def test_runset_selects_baseline_and_exactly_three_ablation_runs(self):
        selected = ablation_runset()
        self.assertEqual(set(selected.run_settings), set(ABLATION_RUN_IDS))
        self.assertEqual(
            set(ABLATION_RUN_IDS),
            {"k2evocuw", "mkgwngn1", "mkgwnex1", "mkgwnrt1"},
        )
        self.assertEqual(selected.filters, "")
        self.assertNotIn("id", selected.filters)

    def test_ablation_panels_are_real_eval_curves_on_epoch_axis(self):
        section = ablation_blocks()
        grids = [block for block in section if isinstance(block, wr.PanelGrid)]
        self.assertEqual(len(grids), 1)
        plots = grids[0].panels
        self.assertEqual(
            {plot.title for plot in plots},
            {"MKG-W ablation MRR", "MKG-W ablation Hits@1", "MKG-W ablation Hits@3", "MKG-W ablation Hits@10"},
        )
        self.assertTrue(all(plot.x == "epoch" for plot in plots))
        self.assertTrue(all(plot.y[0].startswith("eval/") for plot in plots))
        self.assertFalse(any("loss" in plot.title.lower() for plot in plots))

    def test_major_sections_include_ablation_before_provenance(self):
        headings = [block.text for block in blocks() if isinstance(block, wr.H1)]
        self.assertEqual(
            headings,
            [
                "Reproduction Overview",
                "Experimental Setup",
                "MKG-W Main Result",
                "MKG-Y Main Result",
                "Cross-Dataset Comparison",
                "Ablation Studies — MKG-W",
                "Reproducibility & Provenance",
            ],
        )

    def test_comparison_text_names_proposed_baseline_and_all_variants(self):
        text = "\n".join(block.text for block in ablation_blocks() if isinstance(block, wr.MarkdownBlock))
        self.assertIn("Full Model (Proposed Method)", text)
        for variant in ("no_gating_noise", "no_exid", "no_relation_temperature"):
            self.assertIn(variant, text)
        self.assertIn("legacy_best_per_metric", text)
        self.assertIn("one seed", text.lower())
        self.assertIn("operational ablation", text.lower())
        for value in (
            "0.3503000000", "0.2983000000", "0.3728000000", "0.4543000000",
            "0.3509000000", "0.2968000000", "0.3724000000", "0.4586000000",
            "0.3586000000", "0.3059000000", "0.3759000000", "0.4628000000",
            "-0.0011488296", "+0.0053945718", "+0.0071511704",
        ):
            self.assertIn(value, text)

    def test_mkgy_training_loss_panel_declaration_is_unchanged(self):
        source = Path("scripts/update_momok_main_wandb_report.py").read_text()
        self.assertIn('wr.LinePlot(title="MKG-Y training loss", x="epoch", y=["train/loss"], title_x="Epoch")', source)
        self.assertNotRegex(source, r'Metric\("id"\)|Metric\(\s*[\'"]id')
        backup = json.loads(Path("wandb_report/backups/MOMOK_MAIN_REPORT_PRE_ABLATIONS.json").read_text())
        original_panel = None

        def find_panel(value):
            nonlocal original_panel
            if isinstance(value, dict):
                config = value.get("config", {})
                if isinstance(config, dict) and config.get("chartTitle") == "MKG-Y training loss":
                    original_panel = config
                for child in value.values():
                    find_panel(child)
            elif isinstance(value, list):
                for child in value:
                    find_panel(child)

        find_panel(backup["report_spec"])
        self.assertEqual(
            original_panel,
            {"chartTitle": "MKG-Y training loss", "xAxis": "epoch", "metrics": ["train/loss"], "xAxisTitle": "Epoch"},
        )


if __name__ == "__main__":
    unittest.main()
