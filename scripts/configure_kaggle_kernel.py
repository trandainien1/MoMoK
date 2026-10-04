#!/usr/bin/env python3
"""Render safe Kaggle metadata from the local, non-secret environment."""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle" / "momok-db15k"
TEMPLATE = KERNEL_DIR / "kernel-metadata.template.json"
OUTPUT = KERNEL_DIR / "kernel-metadata.json"


def main() -> None:
    username = os.environ.get("KAGGLE_USERNAME_SLUG", "").strip()
    slug = os.environ.get("KAGGLE_KERNEL_SLUG", "momok-mkg-w-reproduction").strip()
    accelerator = os.environ.get("KAGGLE_ACCELERATOR", "NvidiaTeslaT4").strip()
    resume_dataset = os.environ.get("KAGGLE_RESUME_DATASET_REF", "").strip()
    if not username or username == "your-kaggle-username":
        raise SystemExit("KAGGLE_USERNAME_SLUG is required")
    if not slug:
        raise SystemExit("KAGGLE_KERNEL_SLUG is required")
    if not accelerator:
        raise SystemExit("KAGGLE_ACCELERATOR must not be empty")

    metadata = json.loads(TEMPLATE.read_text())
    metadata["id"] = f"{username}/{slug}"
    metadata["machine_shape"] = accelerator
    metadata["dataset_sources"] = [resume_dataset] if resume_dataset else []
    OUTPUT.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"KERNEL_METADATA=PASS")
    print(f"KERNEL={metadata['id']}")
    print(f"ACCELERATOR={accelerator}")


if __name__ == "__main__":
    main()
