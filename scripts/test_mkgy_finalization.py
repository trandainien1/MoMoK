#!/usr/bin/env python3
"""Schema regressions for the post-training MKG-Y W&B finalizer."""

from __future__ import annotations

import json

from finalize_momok_mkgy_wandb import build_comparison_rows, build_environment_rows, validate_table_rows


def main() -> None:
    summary = json.loads(open("kaggle_outputs/mkg-y/latest_ui/repro_output/repro_summary.json").read())
    manifest = json.loads(open("kaggle_outputs/mkg-y/latest_ui/repro_output/dataset_manifest.json").read())
    environment = json.loads(open("kaggle_outputs/mkg-y/latest_ui/repro_output/environment.json").read())
    comparison = build_comparison_rows()
    env_rows = build_environment_rows(summary, manifest, environment)
    validate_table_rows(comparison, env_rows)
    print("ENV_TABLE_SCHEMA_TEST=PASS")
    print("COMPARISON_TABLE_SCHEMA_TEST=PASS")


if __name__ == "__main__":
    main()
