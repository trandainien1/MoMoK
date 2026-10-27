#!/usr/bin/env python3
"""Publish one already-completed MKG-W ablation to its stable W&B run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import wandb as _wandb
except ImportError:  # Allows the publication contract to be tested with a fake adapter.
    _wandb = None

from ablations.wandb_publication import (
    ENTITY,
    GROUP,
    JOB_TYPE,
    PROJECT,
    PUBLICATION_VARIANTS,
    VariantPublicationError,
    build_environment_rows,
    build_history_records,
    build_variant_summary,
    comparison_table_rows,
    existing_run_action,
    publication_for,
    validate_safe_output_dir,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_files(export_dir: Path, evidence_dir: Path) -> list[tuple[Path, str]]:
    pairs = [
        (export_dir / "history.csv", "history.csv"),
        (export_dir / "history_summary.json", "history_summary.json"),
        (export_dir / "audited_final_metrics.json", "audited_final_metrics.json"),
        (export_dir / "provenance.json", "provenance.json"),
    ]
    pairs.extend(
        (evidence_dir / filename, filename)
        for filename in (
            "ablation_smoke.log",
            "dataset_manifest.json",
            "output_inventory.json",
            "checkpoint_final.pt.sha256",
        )
    )
    for source, _ in pairs:
        if not source.is_file():
            raise VariantPublicationError(f"required safe evidence file is missing: {source.name}")
    forbidden = re.compile(rb"(?:WANDB_API_KEY|KAGGLE_API_TOKEN|GH_TOKEN|GITHUB_TOKEN)\s*=|accessToken=|github_pat_|ghp_[A-Za-z0-9_]{20,}")
    for source, _ in pairs:
        data = source.read_bytes()
        if forbidden.search(data):
            raise VariantPublicationError(f"credential-like content found in evidence file {source.name}")
    return pairs


def _fingerprint(files: list[tuple[Path, str]]) -> str:
    rows = [{"name": name, "sha256": _sha256(path)} for path, name in files]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_payload(variant_id: str, export_dir: Path, evidence_dir: Path):
    item = publication_for(variant_id)
    validate_safe_output_dir(evidence_dir, item)
    with (export_dir / "history.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    records = build_history_records(csv_rows)
    if len(records) != 20 or [record["epoch"] for record in records] != list(range(100, 2001, 100)):
        raise VariantPublicationError("publication requires 20 unique observed epochs from 100 through 2000")
    evaluations = [record for record in records if "eval/MRR" in record]
    if len(evaluations) != 20 or evaluations[-1]["epoch"] != 2000:
        raise VariantPublicationError("publication requires 20 observed evaluations, ending at epoch 2000")

    history_summary = _read_json(export_dir / "history_summary.json")
    if history_summary.get("fabricated_points") != 0 or history_summary.get("evaluation_points") != 20:
        raise VariantPublicationError("history summary does not confirm 20 observed evaluations and zero fabricated points")
    final_metrics = history_summary.get("legacy_final_metrics", {})
    for metric, expected in item.final_metrics.items():
        if abs(float(final_metrics.get(metric, float("nan"))) - expected) > 5e-5:
            raise VariantPublicationError(f"audited terminal {metric} does not match the publication registry")
    if history_summary.get("metric_policy") != item.metric_policy:
        raise VariantPublicationError("history metric policy is missing or inconsistent")

    sidecar = evidence_dir / "checkpoint_final.pt.sha256"
    digest = sidecar.read_text(encoding="utf-8").split()[0]
    final_summary = build_variant_summary(item, digest)
    files = _export_files(export_dir, evidence_dir)
    environment = {"gpu": "NVIDIA Tesla T4"}
    config = {
        "paper_title": "Multiple Heads Are Better Than One: Mixture of Modality Knowledge Experts for Entity Representation Learning",
        "paper_venue": "ICLR 2025",
        "task": "MMKGC",
        "dataset": "MKG-W",
        "dataset_fingerprint": {
            "entities": 15000, "relations": 169, "train": 34196,
            "valid": 4276, "test": 4274,
            "image_shape": [15000, 383], "text_shape": [15000, 384],
        },
        "variant": variant_id,
        "variant_definition": item.operational_definition,
        "operational_ablation": True,
        "seed": 10010,
        "lr": 0.001,
        "mu": item.config["mu"],
        "dim": 200,
        "r_dim": 256,
        "batch_size": 1024,
        "n_exp": 3,
        "epochs": 2000,
        "eval_freq": 100,
        "source_repository": "https://github.com/zjukg/MoMoK",
        "source_commit": item.source_commit,
        "kaggle_kernel": f"nientrandai/{item.kernel_slug}",
        "kaggle_version": item.kaggle_version,
        "effective_gpu": environment["gpu"],
        "tracking_mode": "retrospective_import_from_kaggle_logs",
        "live_tracking": False,
        "metric_policy": item.metric_policy,
        "final_metric_source": "terminal_Test_set_results_independent_best_per_metric",
        "baseline_run_id": item.baseline_run_id,
        "baseline_label": item.baseline_label,
        **item.config,
    }
    return item, records, history_summary, final_summary, config, environment, files


def _summary_dict(run) -> dict:
    value = run.summary
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "_json_dict"):
        return dict(value._json_dict)
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def _existing_run(api, run_path: str):
    try:
        return api.run(run_path)
    except Exception as exc:  # W&B SDK error classes differ across releases.
        message = str(exc).lower()
        if "not found" in message or "404" in message or "does not exist" in message:
            return None
        raise


def publish_variant(variant_id: str, export_dir: Path, evidence_dir: Path, wandb_module=None) -> dict:
    if wandb_module is None:
        if _wandb is None:
            raise RuntimeError("install the W&B SDK to publish an ablation run")
        wandb_module = _wandb
    item, records, history_summary, final_summary, config, environment, files = _load_payload(
        variant_id, Path(export_dir), Path(evidence_dir)
    )
    fingerprint = _fingerprint(files)
    run_path = f"{ENTITY}/{PROJECT}/{item.run_id}"
    api = wandb_module.Api()
    existing = _existing_run(api, run_path)
    action = "new"
    if existing is not None:
        existing_config = dict(getattr(existing, "config", {}) or {})
        for key, expected in (("variant", variant_id), ("dataset", "MKG-W"), ("source_commit", item.source_commit)):
            if existing_config.get(key) not in (None, expected):
                raise VariantPublicationError(f"existing W&B run ID belongs to a different {key}")
        existing_history = list(existing.scan_history(keys=["epoch"]))
        action = existing_run_action(_summary_dict(existing), existing_history, fingerprint)
        if action == "skip_complete":
            return {"run_id": item.run_id, "run_url": existing.url, "skipped": True, "fingerprint": fingerprint}

    run = wandb_module.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type=JOB_TYPE,
        name=f"momok-mkgw-{variant_id.replace('_', '-')}-kaggle-v1-seed10010",
        id=item.run_id,
        resume="allow",
        tags=["MoMoK", "ICLR-2025", "MKG-W", "ablation", variant_id, "Kaggle", "Kaggle-v1", "2000-epochs", "retrospective-import", "operational-ablation"],
        config=config,
    )
    if run is None or run.id != item.run_id:
        raise VariantPublicationError("W&B initialized an unexpected run ID; publication stopped")
    if action == "resume_empty":
        existing_summary = _summary_dict(existing)
        if existing_summary or existing_history:
            run.finish(exit_code=1)
            raise VariantPublicationError("refusing to append to a nonempty incomplete W&B run")

    try:
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("eval/*", step_metric="epoch")
        run.define_metric("runtime/*", step_metric="epoch")
        for record in records:
            run.log(record)

        table = wandb_module.Table(
            columns=["Metric", "Full Model (Proposed Method)", "Ablation", "Signed Delta", "Absolute Delta"],
            data=comparison_table_rows(item),
        )
        run.log({"ablation_vs_full_model": table})
        env_table = wandb_module.Table(
            columns=["Key", "Value"],
            data=build_environment_rows(item, environment),
        )
        run.log({"reproduction_environment": env_table})

        artifact_name = f"momok-mkg-w-{variant_id.replace('_', '-')}-kaggle-v1-evidence"
        artifact = wandb_module.Artifact(
            artifact_name,
            type="ablation-evidence",
            metadata={
                "dataset": "MKG-W",
                "variant": variant_id,
                "completed_epoch": 2000,
                "source_commit": item.source_commit,
                "kaggle_kernel": f"nientrandai/{item.kernel_slug}",
                "kaggle_version": item.kaggle_version,
                "checkpoint_filename": "checkpoint_final.pt",
                "checkpoint_sha256": item.checkpoint_sha256,
                "checkpoint_uploaded": False,
                "metric_policy": item.metric_policy,
                "publication_fingerprint": fingerprint,
            },
        )
        for source, name in files:
            artifact.add_file(str(source), name=name)
        run.log_artifact(artifact)

        final_summary.update(
            {
                "publication_fingerprint": fingerprint,
                "publication_tables_written": True,
                "publication_artifact_written": True,
                "publication_complete": True,
                "publication_artifact_name": artifact_name,
                "observed_training_points": history_summary["training_points"],
                "observed_evaluation_points": history_summary["evaluation_points"],
                "first_observed_epoch": history_summary["first_epoch_found"],
                "last_observed_epoch": history_summary["last_epoch_found"],
                "fabricated_points": 0,
            }
        )
        run.summary.update(final_summary)
        run.finish()
    except Exception:
        run.summary.update({"publication_partial": True, "publication_fingerprint": fingerprint})
        run.finish(exit_code=1)
        raise
    return {"run_id": run.id, "run_url": run.url, "skipped": False, "fingerprint": fingerprint}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(PUBLICATION_VARIANTS), required=True)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    for name in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"):
        if not os.environ.get(name):
            raise SystemExit(f"{name} must be set in the local process environment")
    item = publication_for(args.variant)
    export_dir = args.export_dir or Path("wandb_export/ablations") / item.variant_id
    evidence_dir = args.evidence_dir or Path("kaggle_outputs/ablations/recovered") / item.kernel_slug
    result = publish_variant(args.variant, export_dir, evidence_dir)
    publication = {
        "variant": args.variant,
        "run_id": result["run_id"],
        "run_url": result["run_url"],
        "entity": ENTITY,
        "project": PROJECT,
        "group": GROUP,
        "tracking_mode": "retrospective_import_from_kaggle_logs",
        "metric_policy": item.metric_policy,
        "skipped_idempotently": result["skipped"],
        "publication_fingerprint": result["fingerprint"],
        "artifact_name": f"momok-mkg-w-{args.variant.replace('_', '-')}-kaggle-v1-evidence",
    }
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "wandb_publication.json").write_text(json.dumps(publication, indent=2) + "\n")
    print(f"WANDB_ABLATION_PUBLISHED=YES VARIANT={args.variant} RUN_ID={result['run_id']}")
    print(f"WANDB_RUN_URL={result['run_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
