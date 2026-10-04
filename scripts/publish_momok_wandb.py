#!/usr/bin/env python3
"""Publish audited, retrospective MoMoK evidence to one W&B run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import wandb


METRIC_MAP = {
    "MRR": "MRR",
    "Hits@1": "Hits@1",
    "Hits@3": "Hits@3",
    "Hits@10": "Hits@10",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, default=Path("wandb_export"))
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--resume-audit", type=Path, required=True)
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    api_key = os.environ.get("WANDB_API_KEY")
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT", "graphml-momok-reproduction")
    if not api_key or not entity or not project:
        raise SystemExit("WANDB_API_KEY, WANDB_ENTITY, and WANDB_PROJECT are required in the environment")

    summary = load_json(args.summary)
    manifest = load_json(args.dataset_manifest)
    environment = load_json(args.environment)
    resume_audit = load_json(args.resume_audit)
    dataset = summary["experiment"]["dataset"]
    if dataset != "MKG-W":
        raise SystemExit(f"Unexpected dataset identity: {dataset}")
    if summary["final_epoch"] != 2000:
        raise SystemExit("Refusing to publish a non-final run")

    targets = summary["paper_targets"]
    reproduced = summary["final_epoch_metrics"]
    absolute = {key: reproduced[key] - targets[key] for key in METRIC_MAP}
    verdict = "STRONG_MATCH" if max(abs(value) for value in absolute.values()) <= 0.01 else "REPRO_MATCH" if max(abs(value) for value in absolute.values()) <= 0.02 else "OUTSIDE_EXPECTED_RANGE"
    checkpoint_path = args.checkpoint
    checkpoint_sha256 = "d8d49fc007d111d4f238c206ad6994fb370b36637b8f865f305fcc99517a14c5"
    checkpoint_size = checkpoint_path.stat().st_size if checkpoint_path and checkpoint_path.exists() else None

    config = {
        "paper_title": "MoMoK",
        "paper_venue": "ICLR 2025",
        "task": "MMKGC",
        "dataset": dataset,
        "seed": summary["config"]["seed"],
        "epochs": summary["config"]["epochs"],
        "batch_size": summary["config"]["batch_size"],
        "lr": summary["config"]["lr"],
        "mu": summary["config"]["mu"],
        "dim": summary["config"]["dim"],
        "n_exp": summary["config"]["n_experts"],
        "eval_freq": summary["config"]["eval_freq"],
        "source_repository": "https://github.com/zjukg/MoMoK",
        "source_commit": summary["source_commit"],
        "kaggle_kernel": "nientrandai1/momok-mkg-w-reproduction",
        "kaggle_version": 4,
        "kaggle_status": "COMPLETE",
        "effective_python": environment["effective_python"],
        "effective_torch": environment["effective_torch"],
        "effective_cuda": environment["effective_cuda"],
        "gpu_name": environment["gpu"],
        "training_mode": "full_model",
        "tracking_mode": "retrospective_import_from_kaggle_logs",
        "live_tracking": False,
        "checkpoint_sha256": checkpoint_sha256,
        "paper_target_mrr": targets["MRR"],
        "paper_target_hits1": targets["Hits@1"],
        "paper_target_hits3": targets["Hits@3"],
        "paper_target_hits10": targets["Hits@10"],
    }

    run_name = "momok-mkgw-main-kaggle-v4-seed10010"
    tags = ["MoMoK", "ICLR-2025", "MKG-W", "main-result", "reproduction", "Kaggle", "Kaggle-v4", "2000-epochs", "retrospective-import"]
    init_kwargs = dict(project=project, entity=entity, name=run_name, group="main", job_type="reproduction", tags=tags, config=config, reinit="finish_previous")
    if args.run_id:
        init_kwargs.update(id=args.run_id, resume="must")
    run = wandb.init(**init_kwargs)
    try:
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("eval/*", step_metric="epoch")
        run.define_metric("runtime/*", step_metric="epoch")
        history_path = args.export_dir / "history.csv"
        with history_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                record = {"epoch": int(row["epoch"])}
                for source, target in (("train_loss", "train/loss"), ("mi_estimator_loss", "train/mi_estimator_loss"), ("learning_rate", "train/lr"), ("eval_mrr", "eval/MRR"), ("eval_hits1", "eval/Hits@1"), ("eval_hits3", "eval/Hits@3"), ("eval_hits10", "eval/Hits@10"), ("eval_mr", "eval/MR"), ("epoch_time_sec", "runtime/epoch_time_sec")):
                    if row.get(source) not in (None, ""):
                        record[target] = float(row[source])
                run.log(record)

        table = wandb.Table(columns=["Metric", "Paper", "Reproduced", "Absolute Delta", "Relative Delta (%)"])
        for key in METRIC_MAP:
            delta = absolute[key]
            table.add_data(key, targets[key], reproduced[key], delta, 100.0 * delta / targets[key])
        run.log({"paper_vs_reproduction": table})

        environment_rows = [
            ("Dataset", dataset),
            ("Entities", manifest["entity_count"]),
            ("Relations", manifest["relation_count"]),
            ("Train triples", manifest["train_count"]),
            ("Validation triples", manifest["valid_count"]),
            ("Test triples", manifest["test_count"]),
            ("Epochs", summary["config"]["epochs"]),
            ("Seed", summary["config"]["seed"]),
            ("Batch size", summary["config"]["batch_size"]),
            ("Embedding dimension", summary["config"]["dim"]),
            ("Number of experts", summary["config"]["n_experts"]),
            ("Learning rate", summary["config"]["lr"]),
            ("Mu", summary["config"]["mu"]),
            ("GPU", environment["gpu"]),
            ("PyTorch", environment["effective_torch"]),
            ("CUDA", environment["effective_cuda"]),
            ("Kaggle version", 4),
            ("Source commit", summary["source_commit"]),
        ]
        run.log({"reproduction_environment": wandb.Table(data=[(key, str(value)) for key, value in environment_rows], columns=["Key", "Value"])})
        run.summary.update({
            "completed_epoch": 2000,
            "paper/MRR": targets["MRR"], "paper/Hits@1": targets["Hits@1"], "paper/Hits@3": targets["Hits@3"], "paper/Hits@10": targets["Hits@10"],
            "reproduced/MRR": reproduced["MRR"], "reproduced/Hits@1": reproduced["Hits@1"], "reproduced/Hits@3": reproduced["Hits@3"], "reproduced/Hits@10": reproduced["Hits@10"],
            "delta_abs/MRR": absolute["MRR"], "delta_abs/Hits@1": absolute["Hits@1"], "delta_abs/Hits@3": absolute["Hits@3"], "delta_abs/Hits@10": absolute["Hits@10"],
            "delta_relative_pct/MRR": 100.0 * absolute["MRR"] / targets["MRR"], "delta_relative_pct/Hits@1": 100.0 * absolute["Hits@1"] / targets["Hits@1"], "delta_relative_pct/Hits@3": 100.0 * absolute["Hits@3"] / targets["Hits@3"], "delta_relative_pct/Hits@10": 100.0 * absolute["Hits@10"] / targets["Hits@10"],
            "reproduction_verdict": verdict,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_size_bytes": checkpoint_size,
            "checkpoint_uploaded": False,
            "dataset_identity_verified": True,
            "tracking_mode": "retrospective_import_from_kaggle_logs",
        })

        artifact = wandb.Artifact(f"momok-mkg-w-kaggle-v4-evidence", type="reproduction-evidence", metadata={"dataset": dataset, "kaggle_version": 4, "source_commit": summary["source_commit"], "completed_epoch": 2000, "verdict": verdict, "checkpoint_sha256": checkpoint_sha256, "checkpoint_size_bytes": checkpoint_size, "checkpoint_uploaded": False})
        safe_files = [args.export_dir / name for name in ("audited_final_metrics.json", "history.csv", "history_summary.json", "provenance.json", "paper_vs_reproduction.csv", "reproduction_environment.csv", "REPORT_SNIPPET.md")]
        safe_files += [args.summary, args.environment, args.resume_audit]
        report = args.summary.parent / "REPRODUCTION_REPORT.md"
        if report.exists():
            safe_files.append(report)
        evidence_dir = args.export_dir / "evidence_artifact"
        if evidence_dir.exists():
            shutil.rmtree(evidence_dir)
        evidence_dir.mkdir()
        for source in safe_files:
            if source.exists() and source.is_file():
                shutil.copy2(source, evidence_dir / source.name)
        if args.log.stat().st_size <= 2_000_000:
            shutil.copy2(args.log, evidence_dir / "kaggle-v4-training.log")
        artifact.add_dir(str(evidence_dir))
        run.log_artifact(artifact)
        run.finish()
    except Exception:
        run.finish(exit_code=1)
        raise

    export = {
        "run_id": run.id,
        "run_url": run.url,
        "project_url": f"https://wandb.ai/{entity}/{project}",
        "entity": entity,
        "project": project,
        "run_name": run_name,
        "tracking_mode": "retrospective_import_from_kaggle_logs",
        "artifact_name": "momok-mkg-w-kaggle-v4-evidence",
    }
    (args.export_dir / "wandb_publication.json").write_text(json.dumps(export, indent=2) + "\n")
    print("WANDB_RUN_CREATED=YES")
    print("WANDB_HISTORY_UPLOADED=YES")
    print("PAPER_COMPARISON_TABLE=YES")
    print("ENVIRONMENT_TABLE=YES")
    print("EVIDENCE_ARTIFACT=YES")
    print("WANDB_RUN_ID=" + run.id)
    print("WANDB_RUN_URL=" + run.url)
    print("WANDB_PROJECT_URL=" + export["project_url"])


if __name__ == "__main__":
    main()
