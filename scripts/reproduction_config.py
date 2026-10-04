"""Central configuration for the active MoMoK reproduction target."""

from __future__ import annotations

DATASET = "MKG-W"
CONFIG_SOURCE = "OFFICIAL_README"
UPSTREAM_COMMIT = "99c2df114d48c79708ea0608644b685183d81bd9"
GOOGLE_DRIVE_FILE_ID = "1dKJdJunb11kDtFr5NLfPlFknS7cRdm9W"
ACTIVE_KERNEL_REF = "nientrandai1/momok-mkg-w-reproduction"


def feature_archive_members(dataset: str = DATASET) -> dict[str, str]:
    """Return exact official archive members for the active dataset."""
    root = f"embeddings/{dataset}"
    return {
        "img_features.pth": f"{root}/img_features.pth",
        "text_features.pth": f"{root}/text_features.pth",
    }

PAPER_TARGETS = {
    "MRR": 0.3589,
    "Hits@1": 0.3038,
    "Hits@3": 0.3754,
    "Hits@10": 0.4613,
}

EXPECTED_DATASET = {
    "entities": 15000,
    "relations": 169,
    "train": 34196,
    "valid": 4276,
    "test": 4274,
    "image_available": 14463,
    "image_dim": 383,
    "text_available": 14123,
    "text_dim": 384,
}

TRAINING_CONFIG = {
    "seed": 10010,
    "lr": 0.001,
    "mu": 0.0001,
    "dim": 200,
    "r_dim": 256,
    "batch_size": 1024,
    "n_exp": 3,
    "epochs": 2000,
    "eval_freq": 100,
    "save": 1,
    "gamma": 1.0,
    "weight_decay": 0,
    "cuda": 0,
}
