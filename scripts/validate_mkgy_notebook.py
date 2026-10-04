#!/usr/bin/env python3
"""Static validation for the MKG-Y Kaggle notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "kaggle" / "momok-mkg-y" / "momok_mkgy_reproduction.ipynb"


def main() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in nb.cells)
    checks = {
        "NOTEBOOK_JSON_VALID": True,
        "ACTIVE_DATASET_MKG_Y": 'DATASET = "MKG-Y"' in source,
        "CONFIG_SOURCE_PRESENT": 'CONFIG_SOURCE = "OFFICIAL_README_PLUS_PINNED_DEFAULTS"' in source,
        "MKGY_TARGETS_PRESENT": all(value in source for value in ("0.3791", "0.3509", "0.392", "0.432")),
        "MKGY_COUNTS_PRESENT": all(value in source for value in ("'relations': 28", "21310", "2665", "2663")),
        "NO_ACTIVE_MKG_W": 'DATASET = "MKG-W"' not in source,
        "NO_ACTIVE_DB15K": 'DATASET = "DB15K"' not in source,
        "EXACT_FEATURE_SELECTION": "embeddings/MKG-Y/img_features.pth" in source and "embeddings/MKG-Y/text_features.pth" in source,
        "WANDB_SECRET_RUNTIME_ONLY": 'get_secret("WANDB_API_KEY")' in source and "WANDB_API_KEY=" not in source,
        "WANDB_LIVE_GATE": 'WANDB_LIVE=YES' in source and "HUMAN_GATE_REQUIRED" in source,
        "WANDB_RUN_NAME": "momok-mkgy-main-kaggle-v1-seed10010" in source,
        "WANDB_STABLE_RUN_ID": "WANDB_RUN_ID =" in source and "id=WANDB_RUN_ID" in source and 'resume="allow"' in source,
        "WANDB_RUN_ID_CHECKPOINTED": "MOMOK_WANDB_RUN_ID" in source and "Checkpoint W&B run ID mismatch" in source,
        "LIVE_TRAINING_EPOCH_MARKER": "TRAIN_EPOCH_METRICS_JSON" in source,
        "WANDB_ARTIFACT": "momok-mkg-y-kaggle-evidence" in source and "paper_vs_reproduction" in source,
        "FULL_COMMAND_CONFIG": all(value in source for value in ('"--lr", str(LR)', '"--dim", str(DIM)', '"--dataset", DATASET', '"--epochs", str(EPOCHS)')),
        "CHECKPOINT_CONTRACT": all(key in source for key in ("model_state_dict", "estimator_state_dict", "optimizer_state_dict", "optimizer_mi_state_dict", "lr_scheduler_state_dict", "torch_cuda_rng_state_all", "corpus_train_indices")),
        "RESUME_GUARD": "Checkpoint config mismatch" in source and "Checkpoint source commit mismatch" in source,
        "SOFT_LIMIT": "SESSION_SOFT_LIMIT_SECONDS = 37800" in source,
        "FINAL_EPOCH": "completed_epoch == 2000" in source or "EPOCHS = 2000" in source,
    }
    parse_ok = True
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            try:
                ast.parse(cell.source, filename=f"mkgy-cell-{index}")
            except SyntaxError as exc:
                parse_ok = False
                print(f"SYNTAX_ERROR_CELL={index}:{exc}")
    checks["CODE_CELLS_PARSE"] = parse_ok
    for name, passed in checks.items():
        print(f"{name}={'YES' if passed else 'NO'}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
