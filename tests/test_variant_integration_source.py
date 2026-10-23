from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]


class VariantIntegrationSourceTests(unittest.TestCase):
    def test_model_source_exposes_variant_switches(self):
        source = (ROOT / "models" / "RelMoE.py").read_text()

        self.assertIn("use_relation_temperature", source)
        self.assertIn("use_gating_noise", source)
        self.assertIn("fusion_mode", source)
        self.assertIn("training_head_names", source)
        self.assertIn("evaluation_head_indices", source)

    def test_training_source_resolves_variant_and_gates_estimator(self):
        source = (ROOT / "train.py").read_text()

        self.assertIn("resolve_variant", source)
        self.assertIn("args.variant_config", source)
        self.assertIn("estimator_should_run", source)

    def test_evaluator_uses_model_selected_heads(self):
        source = (ROOT / "utils" / "data_loader.py").read_text()

        self.assertIn("evaluation_head_indices", source)


if __name__ == "__main__":
    unittest.main()
