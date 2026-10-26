#!/usr/bin/env python3
"""Export only metrics observed in a completed MoMoK ablation stdout log."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ablations.wandb_publication import publication_for


_NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
_METRIC_KEYS = {
    "test_MRR": "MRR",
    "test_Hits@1": "Hits@1",
    "test_Hits@3": "Hits@3",
    "test_Hits@10": "Hits@10",
    "test_MR": "MR",
}


def _field(text: str, name: str) -> str | None:
    match = re.search(rf"(?:^|[,( ]){re.escape(name)}=([^, )]+)", text)
    return match.group(1).strip("'\"") if match else None


def _metrics(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for source_key, output_key in _METRIC_KEYS.items():
        match = re.search(rf"\b{re.escape(source_key)}:\s*{_NUMBER}", text)
        if match:
            result[output_key] = float(match.group(1))
    return result


def parse_raw_ablation_log(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Parse observed epoch events and the separate terminal legacy summary.

    The epoch history represents test evaluations at their logged epochs.
    The final comparison metrics come from the last ``Test set results``
    section, which upstream reports as independently best-per-metric values.
    """

    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    config_line = next((line for line in text.splitlines() if line.startswith("Namespace(")), "")
    if not config_line:
        raise ValueError("training log has no Namespace configuration line")
    config = {
        key: _field(config_line, key)
        for key in ("dataset", "variant", "lr", "mu", "dim", "r_dim", "epochs", "batch_size", "n_exp", "seed", "eval_freq")
    }
    config = {key: value for key, value in config.items() if value is not None}
    variant_id = config.get("variant")
    if not variant_id:
        raise ValueError("training log has no variant configuration")
    item = publication_for(variant_id)

    losses: dict[int, dict[str, float]] = {}
    for match in re.finditer(
        rf"Epoch\s+(\d+)\s*,\s*average loss\s+{_NUMBER}(?:\s*,\s*epoch_time\s+{_NUMBER})?",
        text,
    ):
        epoch = int(match.group(1))
        values = {"train_main_loss": float(match.group(2))}
        if match.group(3) is not None:
            values["epoch_time_sec"] = float(match.group(3))
        losses[epoch] = values

    # tqdm emits carriage-return delimited progress records. Keep the last
    # actual observed MI estimate for each displayed epoch; never aggregate it.
    progress_values: dict[int, float] = {}
    normalized = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).replace("\r", "\n")
    progress_pattern = re.compile(
        rf"(?:^|\s)(\d+)/\d+[^\n]*?loss=main:\s*{_NUMBER}\s+mi:\s*{_NUMBER}",
        re.MULTILINE,
    )
    for match in progress_pattern.finditer(normalized):
        epoch = int(match.group(1))
        progress_values[epoch] = float(match.group(3))

    evaluations: dict[int, dict[str, float]] = {}
    for match in re.finditer(r"(?m)^Epoch:\s*(\d+)\s*$", text):
        epoch = int(match.group(1))
        next_epoch = re.search(r"(?m)^Epoch:\s*\d+\s*$", text[match.end():])
        block_end = match.end() + next_epoch.start() if next_epoch else len(text)
        block = text[match.end():block_end]
        metric_line = next((line for line in block.splitlines() if "test_MRR:" in line), "")
        parsed = _metrics(metric_line)
        if {"MRR", "Hits@1", "Hits@3", "Hits@10"}.issubset(parsed):
            evaluations[epoch] = parsed

    terminal_sections = list(re.finditer(r"(?m)^Test set results:\s*$", text))
    if not terminal_sections:
        raise ValueError("training log has no terminal Test set results block")
    terminal = text[terminal_sections[-1].end():]
    terminal_line = next((line for line in terminal.splitlines() if "test_MRR:" in line), "")
    legacy_final = _metrics(terminal_line)
    required_metrics = {"MRR", "Hits@1", "Hits@3", "Hits@10"}
    if not required_metrics.issubset(legacy_final):
        raise ValueError("terminal Test set results block is missing required metrics")
    if 2000 not in evaluations:
        raise ValueError("epoch-2000 evaluation is missing")

    epochs = sorted(set(losses) | set(evaluations))
    rows: list[dict[str, Any]] = []
    for epoch in epochs:
        row: dict[str, Any] = {"epoch": epoch}
        row.update(losses.get(epoch, {}))
        if epoch in progress_values:
            row["mi_estimator_loss"] = progress_values[epoch]
        for key, value in evaluations.get(epoch, {}).items():
            row[f"eval_{key}"] = value
        rows.append(row)

    if config.get("dataset") != "MKG-W":
        raise ValueError(f"expected MKG-W log, found dataset={config.get('dataset')!r}")
    for key, value in item.config.items():
        if key in config and str(config[key]) != str(value):
            raise ValueError(f"{variant_id}: log config {key}={config[key]!r}, expected {value!r}")
    for metric, expected in item.final_metrics.items():
        if abs(legacy_final[metric] - expected) > 5e-5:
            raise ValueError(
                f"{variant_id}: terminal {metric}={legacy_final[metric]}, expected {expected}"
            )

    summary = {
        "variant": variant_id,
        "dataset": "MKG-W",
        "first_epoch_found": min(epochs) if epochs else None,
        "last_epoch_found": max(epochs) if epochs else None,
        "training_points": sum("train_main_loss" in row for row in rows),
        "evaluation_points": len(evaluations),
        "evaluation_epochs": sorted(evaluations),
        "fabricated_points": 0,
        "epoch_2000_metrics": evaluations[2000],
        "legacy_final_metrics": legacy_final,
        "final_metric_source": "terminal_Test_set_results_independent_best_per_metric",
        "metric_policy": item.metric_policy,
        "config": config,
    }
    provenance = {
        "tracking_mode": "retrospective_import_from_kaggle_logs",
        "live_wandb_tracking": False,
        "kaggle_kernel": f"nientrandai/{item.kernel_slug}",
        "kaggle_version": item.kaggle_version,
        "completed_epoch": 2000,
        "dataset_identity": "MKG-W",
        "source_commit": item.source_commit,
        "source_log": str(path.resolve()),
        "parser_version": 1,
        "metric_policy": item.metric_policy,
    }
    return rows, summary, provenance


def write_export(path: Path, output_dir: Path) -> None:
    rows, summary, provenance = parse_raw_ablation_log(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "history.csv"
    columns = ["epoch", "train_main_loss", "mi_estimator_loss", "eval_MRR", "eval_Hits@1", "eval_Hits@3", "eval_Hits@10", "eval_MR", "epoch_time_sec"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "history_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "audited_final_metrics.json").write_text(
        json.dumps(summary["legacy_final_metrics"], indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_export(args.log, args.output_dir)
    _, summary, _ = parse_raw_ablation_log(args.log)
    print(
        f"HISTORY_PARSE=PASS FIRST_EPOCH_FOUND={summary['first_epoch_found']} "
        f"LAST_EPOCH_FOUND={summary['last_epoch_found']} "
        f"TRAIN_POINTS={summary['training_points']} "
        f"EVAL_POINTS={summary['evaluation_points']} FABRICATED_POINTS=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
