#!/usr/bin/env python3
"""Render MKG-Y Kaggle metadata without reading credentials."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle" / "momok-mkg-y"
TEMPLATE = KERNEL_DIR / "kernel-metadata.template.json"
OUTPUT = KERNEL_DIR / "kernel-metadata.json"


def main() -> None:
    owner = os.environ.get("KAGGLE_USERNAME_SLUG", "").strip()
    slug = os.environ.get("KAGGLE_MKGY_KERNEL_SLUG", "momok-mkg-y-reproduction").strip()
    accelerator = os.environ.get("KAGGLE_ACCELERATOR", "NvidiaTeslaA100").strip()
    if owner != "nientrandai1":
        raise SystemExit("KAGGLE_ACCOUNT_MISMATCH=FAIL")
    metadata = json.loads(TEMPLATE.read_text())
    metadata["id"] = f"{owner}/{slug}"
    metadata["machine_shape"] = accelerator
    OUTPUT.write_text(json.dumps(metadata, indent=2) + "\n")
    print("KERNEL_METADATA=PASS")
    print("KERNEL=" + metadata["id"])
    print("ACCELERATOR=" + accelerator)


if __name__ == "__main__":
    main()
