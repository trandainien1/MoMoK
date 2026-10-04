#!/usr/bin/env python3
"""Static validation for the generated MKG-W Kaggle notebook."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import nbformat

from reproduction_config import ACTIVE_KERNEL_REF, DATASET, EXPECTED_DATASET, PAPER_TARGETS, UPSTREAM_COMMIT


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "kaggle" / "momok-db15k" / "momok_db15k_reproduction.ipynb"
PINNED_COMMIT = "99c2df114d48c79708ea0608644b685183d81bd9"


def main() -> None:
    checks: dict[str, bool] = {}
    try:
        nb = nbformat.read(NOTEBOOK, as_version=4)
        checks["NOTEBOOK_JSON_VALID"] = True
    except Exception:
        checks["NOTEBOOK_JSON_VALID"] = False
        raise

    checks["CELL_COUNT_EXPECTED"] = len(nb.cells) == 18
    source = "\n".join(cell.source for cell in nb.cells)
    checks["NO_KAGGLE_SECRET_LITERAL"] = not bool(re.search(r"(?i)(?:api[_-]?token|secret|legacy[_-]?api[_-]?key)\\s*[:=]\\s*['\"][^'\"]{12,}['\"]", source))
    checks["NO_WANDB_SECRET_LITERAL"] = not bool(re.search(r"(?i)WANDB_API_KEY\\s*[:=]\\s*['\"][^'\"]{12,}['\"]", source))
    checks["PINNED_COMMIT_PRESENT"] = PINNED_COMMIT in source
    checks["ACTIVE_DATASET_MKG_W"] = 'DATASET = "MKG-W"' in source
    checks["OFFICIAL_MKG_W_CONFIG_PRESENT"] = 'CONFIG_SOURCE = "OFFICIAL_README"' in source and 'RESUME_FROM_DB15K=NO' in source
    checks["MKG_W_PAPER_TARGETS_PRESENT"] = all(str(value) in source for value in PAPER_TARGETS.values())
    checks["MKG_W_ENTITY_COUNT_EXPECTED"] = str(EXPECTED_DATASET["entities"]) in source
    checks["MKG_W_RELATION_COUNT_EXPECTED"] = str(EXPECTED_DATASET["relations"]) in source
    checks["MKG_W_TRAIN_COUNT_EXPECTED"] = str(EXPECTED_DATASET["train"]) in source
    checks["MKG_W_VALID_COUNT_EXPECTED"] = str(EXPECTED_DATASET["valid"]) in source
    checks["MKG_W_TEST_COUNT_EXPECTED"] = str(EXPECTED_DATASET["test"]) in source
    checks["NO_DB15K_ACTIVE_CONFIG"] = 'DATASET = "DB15K"' not in source and 'PAPER_METRICS = {"MRR": 0.3957' not in source
    checks["GOOGLE_DRIVE_ID_PRESENT"] = "1dKJdJunb11kDtFr5NLfPlFknS7cRdm9W" in source
    checks["FULL_CHECKPOINT_REQUIRED_KEYS"] = all(key in source for key in [
        "format_version", "completed_epoch", "model_state_dict", "estimator_state_dict",
        "optimizer_state_dict", "optimizer_mi_state_dict", "lr_scheduler_state_dict",
        "best_test_metrics", "final_epoch_metrics", "python_random_state",
        "numpy_rng_state", "torch_rng_state", "torch_cuda_rng_state_all",
        "corpus_train_indices", "training_config", "source_commit", "dataset_metadata",
    ])
    checks["RNG_RESTORE_AFTER_CONSTRUCTION"] = (
        source.find("model.load_state_dict") < source.find("random.setstate")
        and source.find("optimizer_mi.load_state_dict") < source.find("torch.set_rng_state")
    )
    checks["END_OF_EPOCH_CHECKPOINT_ORDER"] = "lr_scheduler.step() -> evaluation -> _save_training_checkpoint" in source
    checks["SESSION_SOFT_LIMIT_PRESENT"] = "session_soft_limit_seconds" in source and "37800" in source
    checks["PARTIAL_HANDOFF_STATE_PRESENT"] = "PARTIAL_CHECKPOINT_READY" in source and "checkpoint_handoff.pt" in source
    checks["FINAL_COMPLETE_STATE_PRESENT"] = "FINAL_COMPLETE" in source and "checkpoint_final.pt" in source
    checks["CONFIG_MISMATCH_FAIL_CLOSED"] = "Checkpoint config mismatch" in source and "source commit mismatch" in source
    checks["CHECKPOINT_SHA256_PRESENT"] = ".sha256" in source and "sha256" in source
    checks["NO_PARTIAL_FINAL_VERDICT"] = "No final metrics or reproduction verdict" in source
    checks["FINAL_EPOCH_EXACTLY_2000"] = "'completed_epoch': args.epochs" in source and "EPOCHS = 2000" in source
    checks["FRESH_RESTART_MODE"] = "RESUME_MODE = \"FRESH_RESTART_WITH_CHECKPOINTING\"" in source and "RESUME_FROM_DB15K=NO" in source
    checks["RUNTIME_CHECKPOINT_ROUNDTRIP"] = all(marker in source for marker in [
        "CHECKPOINT_RUNTIME_ROUNDTRIP=PASS", "CHECKPOINT_SHA256_VERIFY=PASS",
        "CHECKPOINT_FULL_STATE_LOAD=PASS",
    ])
    checks["RESUME_CONTROL_FLOW_TEST"] = all(marker in source for marker in [
        "RESUME_NEXT_EPOCH_TEST=PASS", "RESUME_NO_REPLAY=PASS", "RESUME_NO_SKIP=PASS",
    ])
    checks["V3_RUNTIMENET_FORWARD_PRESENT"] = "class _RuntimeNet" in source and "def forward(self, x):" in source and "return self.layer(x)" in source
    checks["ACTIVE_KERNEL_REF_MKG_W"] = ACTIVE_KERNEL_REF in source
    checks["NO_ACTIVE_DB15K_KERNEL_REF"] = "nientrandai1/momok-db15k-reproduction" not in source
    code_cells_ok = True
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            try:
                ast.parse(cell.source, filename=f"cell-{index}")
            except SyntaxError as exc:
                code_cells_ok = False
                print(f"SYNTAX_ERROR_CELL={index}:{exc}")
    checks["CODE_CELLS_PARSE"] = code_cells_ok

    for name, passed in checks.items():
        print(f"{name}={'YES' if passed else 'NO'}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
