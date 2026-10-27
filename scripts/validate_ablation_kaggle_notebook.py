#!/usr/bin/env python3
"""Static validation for the generated ablation notebook."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SOURCE_COMMIT = "99c2df114d48c79708ea0608644b685183d81bd9"


def validate_notebook(path: Path) -> list[str]:
    try:
        notebook = json.loads(Path(path).read_text())
    except Exception as exc:
        return [f"invalid JSON: {exc}"]
    errors = []
    if notebook.get("nbformat") != 4:
        errors.append("nbformat must be 4")
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
    for marker in (SOURCE_COMMIT, "embeddings/{DATASET}/img_features.pth", "embeddings/{DATASET}/text_features.pth", "--variant", "--checkpoint_every", "checkpoint_latest.pt", "CHECKPOINT_ROUNDTRIP=PASS", "RESUME_NEXT_EPOCH=3", "DATASET_VALIDATION=PASS"):
        if marker not in source:
            errors.append("missing marker: " + marker)
    if "WANDB_API_KEY=" in source or "KAGGLE_API_TOKEN=" in source or "GH_TOKEN=" in source:
        errors.append("credential marker embedded")
    if re.search(r"accessToken=|github_pat_|ghp_", source):
        errors.append("credential-like literal embedded")
    return errors


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("kaggle/momok-ablation/momok_ablation_reproduction.ipynb")
    errors = validate_notebook(path)
    if errors:
        for error in errors:
            print("FAIL=" + error)
        return 1
    print("ABLATION_NOTEBOOK_VALIDATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
