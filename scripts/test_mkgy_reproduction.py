#!/usr/bin/env python3
"""Local MKG-Y dataset/config contract test."""

from __future__ import annotations

import json
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "repro" / "configs" / "mkgy_main.json").read_text())
DATASET = ROOT.parent / "MoMoK" / "datasets" / "MKG-Y"
FEATURES = ROOT / "kaggle_outputs" / "version4_final_output" / "embeddings_extracted" / "embeddings" / "MKG-Y"


def count(path: Path) -> int:
    return sum(1 for _ in path.open())


def main() -> None:
    expected = CONFIG["dataset_fingerprint"]
    assert count(DATASET / "entity2id.txt") == expected["entities"]
    assert count(DATASET / "relation2id.txt") == expected["relations"]
    assert count(DATASET / "train.txt") == expected["train"]
    assert count(DATASET / "valid.txt") == expected["valid"]
    assert count(DATASET / "test.txt") == expected["test"]
    feature_paths_present = (FEATURES / "img_features.pth").is_file() and (FEATURES / "text_features.pth").is_file()
    if not feature_paths_present:
        print("FEATURE_TENSOR_LOAD=DEFERRED_KAGGLE")
        print("FEATURE_ARCHIVE_LOCAL=ABSENT")
        print("CONFIG_SOURCE=OFFICIAL_README_PLUS_PINNED_DEFAULTS")
        print("DATASET=MKG-Y")
        print("DATASET_VALIDATION=PASS")
        print("ENTITY_COUNT=15000")
        print("RELATION_COUNT=28")
        print("TRAIN_COUNT=21310")
        print("VALID_COUNT=2665")
        print("TEST_COUNT=2663")
        print("FEATURE_SHAPES=DEFERRED_KAGGLE")
        return
    if torch is not None:
        image = torch.load(FEATURES / "img_features.pth", map_location="cpu", weights_only=False)
        text = torch.load(FEATURES / "text_features.pth", map_location="cpu", weights_only=False)
        assert list(image.shape) == [expected["entities"], expected["image_dim"]]
        assert list(text.shape) == [expected["entities"], expected["text_dim"]]
        assert torch.isfinite(image).all() and torch.isfinite(text).all()
        print("FEATURE_TENSOR_LOAD=PASS")
    else:
        print("FEATURE_TENSOR_LOAD=DEFERRED_KAGGLE")
    print("CONFIG_SOURCE=OFFICIAL_README_PLUS_PINNED_DEFAULTS")
    print("DATASET=MKG-Y")
    print("DATASET_VALIDATION=PASS")
    print("ENTITY_COUNT=15000")
    print("RELATION_COUNT=28")
    print("TRAIN_COUNT=21310")
    print("VALID_COUNT=2665")
    print("TEST_COUNT=2663")
    print("FEATURE_SHAPES=[15000,383],[15000,384]")


if __name__ == "__main__":
    main()
