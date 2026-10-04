#!/usr/bin/env python3
"""Parse metrics actually emitted by a completed MoMoK Kaggle log.

The Kaggle log is a JSON-lines stream whose ``data`` values contain the
original stdout/stderr fragments.  This parser deliberately keeps only
values observed in the full-training segment after FULL_COMMAND and never
interpolates missing epochs or evaluation points.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
LOSS_RE = re.compile(rf"loss=main:\s*({NUMBER})\s+mi:\s*({NUMBER})")
EPOCH_RE = re.compile(rf"Epoch\s+(\d+)\s*,\s*average loss\s+({NUMBER})\s*,\s*epoch_time\s+({NUMBER})")
EVAL_EPOCH_RE = re.compile(r"Epoch:\s*(\d+)")
EVAL_RE = re.compile(
    rf"test_Hits@10:\s*({NUMBER})\s+test_Hits@3:\s*({NUMBER})\s+"
    rf"test_Hits@1:\s*({NUMBER})\s+test_MR:\s*({NUMBER})\s+test_MRR:\s*({NUMBER})"
)


def iter_fragments(path: Path):
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.lstrip(",")
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        data = item.get("data")
        if isinstance(data, str):
            yield item.get("stream_name"), float(item.get("time", 0.0)), data


def parse(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    full_training = False
    current_eval_epoch: int | None = None
    latest_main: float | None = None
    latest_mi: float | None = None
    source_values: dict[str, list[str]] = {}

    for stream, timestamp, fragment in iter_fragments(path):
        if not full_training and "FULL_COMMAND=" in fragment:
            full_training = True
        if not full_training or stream != "stdout":
            continue

        for line in fragment.replace("\r", "\n").splitlines():
            loss_match = LOSS_RE.search(line)
            if loss_match:
                latest_main = float(loss_match.group(1))
                latest_mi = float(loss_match.group(2))

            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                record = records.setdefault(epoch, {"epoch": epoch})
                record["train_loss"] = float(epoch_match.group(2))
                record["mi_estimator_loss"] = latest_mi
                record["epoch_time_sec"] = float(epoch_match.group(3))
                record["_source_timestamp_sec"] = timestamp
                source_values.setdefault("train_loss", []).append(line)

            eval_epoch_match = EVAL_EPOCH_RE.search(line)
            if eval_epoch_match:
                current_eval_epoch = int(eval_epoch_match.group(1))

            eval_match = EVAL_RE.search(line)
            if eval_match and current_eval_epoch is not None and "eval_mrr" not in records.get(current_eval_epoch, {}):
                epoch = current_eval_epoch
                record = records.setdefault(epoch, {"epoch": epoch})
                record.update(
                    {
                        "eval_hits10": float(eval_match.group(1)),
                        "eval_hits3": float(eval_match.group(2)),
                        "eval_hits1": float(eval_match.group(3)),
                        "eval_mr": float(eval_match.group(4)),
                        "eval_mrr": float(eval_match.group(5)),
                    }
                )
                record["_source_timestamp_sec"] = timestamp
                source_values.setdefault("eval_metrics", []).append(line)

    ordered = [records[epoch] for epoch in sorted(records)]
    fields = [
        "epoch",
        "train_loss",
        "mi_estimator_loss",
        "learning_rate",
        "eval_mrr",
        "eval_hits1",
        "eval_hits3",
        "eval_hits10",
        "eval_mr",
        "epoch_time_sec",
    ]
    clean_rows = [{field: record.get(field) for field in fields} for record in ordered]
    summary = {
        "source_log": str(path),
        "parser_version": 1,
        "first_epoch_found": clean_rows[0]["epoch"] if clean_rows else None,
        "last_epoch_found": clean_rows[-1]["epoch"] if clean_rows else None,
        "training_points": sum(row["train_loss"] is not None for row in clean_rows),
        "evaluation_points": sum(row["eval_mrr"] is not None for row in clean_rows),
        "fabricated_points": 0,
        "training_loss_source": "Epoch average loss emitted by MoMoK training log",
        "mi_loss_source": "latest observed progress-bar mi value at each epoch boundary",
        "evaluation_source": "test_* metrics emitted by MoMoK evaluation log",
    }
    return clean_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("wandb_export"))
    args = parser.parse_args()
    rows, summary = parse(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "mi_estimator_loss",
                "learning_rate",
                "eval_mrr",
                "eval_hits1",
                "eval_hits3",
                "eval_hits10",
                "eval_mr",
                "epoch_time_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "history_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance = {
        "tracking_mode": "retrospective_import_from_kaggle_logs",
        "live_wandb_tracking": False,
        "kaggle_version": 4,
        "completed_epoch": 2000,
        "dataset_identity": "MKG-W",
        "source_log": str(args.log),
        "parser_version": 1,
        "source_value_policy": "observed_values_only; no interpolation or synthetic points",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"FIRST_EPOCH_FOUND={summary['first_epoch_found']}")
    print(f"LAST_EPOCH_FOUND={summary['last_epoch_found']}")
    print(f"TRAIN_POINTS={summary['training_points']}")
    print(f"EVAL_POINTS={summary['evaluation_points']}")
    print("FABRICATED_POINTS=0")


if __name__ == "__main__":
    main()
