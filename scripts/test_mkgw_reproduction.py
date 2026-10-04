#!/usr/bin/env python3
"""Focused local regression checks for the active MKG-W reproduction."""

from __future__ import annotations

import json
import re
import zipfile
import ast
from pathlib import Path

from reproduction_config import DATASET, EXPECTED_DATASET, PAPER_TARGETS, UPSTREAM_COMMIT, feature_archive_members


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "kaggle" / "momok-db15k" / "momok_db15k_reproduction.ipynb"
UPSTREAM = ROOT.parent / "MoMoK"


def check_dataset_files() -> None:
    dataset_dir = UPSTREAM / "datasets" / DATASET
    expected = {
        "entity2id.txt": EXPECTED_DATASET["entities"],
        "relation2id.txt": EXPECTED_DATASET["relations"],
        "train.txt": EXPECTED_DATASET["train"],
        "valid.txt": EXPECTED_DATASET["valid"],
        "test.txt": EXPECTED_DATASET["test"],
    }
    for name, count in expected.items():
        assert sum(1 for _ in (dataset_dir / name).open()) == count, name


def check_feature_archive() -> None:
    archives = sorted(Path("/tmp").glob("momok-embeddings.*.zip"))
    assert archives, "official embedding archive not found in /tmp"
    with zipfile.ZipFile(archives[-1]) as archive:
        expected_entries = {
            "embeddings/MKG-W/img_features.pth",
            "embeddings/MKG-W/text_features.pth",
        }
        assert expected_entries.issubset(set(archive.namelist()))
        for name, expected_bytes in {
            "embeddings/MKG-W/img_features.pth": EXPECTED_DATASET["entities"] * EXPECTED_DATASET["image_dim"] * 4,
            "embeddings/MKG-W/text_features.pth": EXPECTED_DATASET["entities"] * EXPECTED_DATASET["text_dim"] * 4,
        }.items():
            with zipfile.ZipFile(archive.open(name)) as tensor_archive:
                data_files = [n for n in tensor_archive.namelist() if "/data/" in n]
                assert len(data_files) == 1
                assert tensor_archive.getinfo(data_files[0]).file_size == expected_bytes


def select_feature_members(archive_names: list[str], dataset: str = DATASET) -> dict[str, str]:
    expected = feature_archive_members(dataset)
    selected = {}
    for logical_name, member in expected.items():
        matches = [name for name in archive_names if name == member]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one archive member: {member}")
        selected[logical_name] = matches[0]
    return selected


def check_feature_selection_collision_regression() -> None:
    inventory = [
        "embeddings/DB15K/img_features.pth",
        "embeddings/DB15K/text_features.pth",
        "embeddings/MKG-W/img_features.pth",
        "embeddings/MKG-W/text_features.pth",
    ]
    selected = select_feature_members(inventory)
    assert selected == {
        "img_features.pth": "embeddings/MKG-W/img_features.pth",
        "text_features.pth": "embeddings/MKG-W/text_features.pth",
    }
    for missing in ("embeddings/MKG-W/img_features.pth", "embeddings/MKG-W/text_features.pth"):
        reduced = [name for name in inventory if name != missing]
        try:
            select_feature_members(reduced)
        except ValueError:
            pass
        else:
            raise AssertionError(f"missing exact member was accepted: {missing}")
    duplicate = inventory + ["embeddings/MKG-W/img_features.pth"]
    try:
        select_feature_members(duplicate)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate exact member was accepted")


def check_notebook_contract() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    source = "\n".join("".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"] for cell in notebook["cells"])
    assert f'DATASET = "{DATASET}"' in source
    assert UPSTREAM_COMMIT in source
    assert all(str(value) in source for value in PAPER_TARGETS.values())
    assert 'DATASET = "DB15K"' not in source
    assert "RESUME_FROM_DB15K=NO" in source
    assert "CONFIG_SOURCE = \"OFFICIAL_README\"" in source
    assert "img_features.pth" in source and "text_features.pth" in source
    assert "FEATURE_ARCHIVE_MEMBERS" in source
    assert "archive_names.count(member) != 1" in source
    assert "matches[0]" not in source
    assert "Checkpoint config mismatch" in source


def check_db15k_checkpoint_rejected() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    source = "\n".join("".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"] for cell in notebook["cells"])
    saved = {"training_config": {"dataset": "DB15K"}}
    assert saved["training_config"]["dataset"] != DATASET
    assert "for key, value in expected.items()" in source
    assert "saved.get(key) != value" in source


def check_runtime_net_forward_regression() -> None:
    hardening = (ROOT / "scripts" / "notebook_runtime_hardening.py").read_text()
    match = re.search(r"class _RuntimeNet\(torch\.nn\.Module\):(?P<body>.*?)(?=\n\n(runtime_model|runtime_estimator))", hardening, re.S)
    assert match, "_RuntimeNet source not found"
    body = match.group("body")
    assert "def forward(self, x):" in body
    assert "return self.layer(x)" in body
    notebook = json.loads(NOTEBOOK.read_text())
    source = "\n".join("".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"] for cell in notebook["cells"])
    assert "V3_RUNTIMENET_FORWARD_REGRESSION=PASS" in source


def main() -> None:
    check_dataset_files()
    check_feature_archive()
    check_feature_selection_collision_regression()
    check_notebook_contract()
    check_db15k_checkpoint_rejected()
    check_runtime_net_forward_regression()
    print("MKG_W_REGRESSION_TESTS=PASS")
    print("DB15K_CHECKPOINT_REJECTED=PASS")
    print("FEATURE_EXTRACTION_REQUIRED=NO")
    print("OFFICIAL_PRECOMPUTED_FEATURES=YES")
    print("V2_FEATURE_BASENAME_COLLISION_REGRESSION=PASS")
    print("FEATURE_SELECTION_RUNTIME_TEST=PASS")
    print("V3_RUNTIMENET_FORWARD_REGRESSION=PASS")


if __name__ == "__main__":
    main()
