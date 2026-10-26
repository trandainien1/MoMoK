"""Immutable provenance and comparison data for completed MKG-W ablations.

This module contains no W&B or Kaggle client code. It is safe to import in
tests and is the single source of truth for retrospective publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


SOURCE_COMMIT = "99c2df114d48c79708ea0608644b685183d81bd9"
ENTITY = "trandainien1"
PROJECT = "graphml-momok-reproduction"
GROUP = "ablations"
JOB_TYPE = "ablation"
BASELINE_RUN_ID = "k2evocuw"
BASELINE_LABEL = "Full Model (Proposed Method)"
BASELINE_METRICS = {
    "MRR": 0.3514488296,
    "Hits@1": 0.2979644361,
    "Hits@3": 0.3711979410,
    "Hits@10": 0.4532054282,
}
METRIC_POLICY = "legacy_best_per_metric"
DATASET_FINGERPRINT = {
    "entities": 15000,
    "relations": 169,
    "train": 34196,
    "valid": 4276,
    "test": 4274,
    "image_shape": (15000, 383),
    "text_shape": (15000, 384),
}
SAFE_OUTPUT_FILES = (
    "ablation_smoke.log",
    "dataset_manifest.json",
    "output_inventory.json",
    "checkpoint_final.pt.sha256",
)


class VariantPublicationError(ValueError):
    """Raised when evidence does not match its frozen publication record."""


@dataclass(frozen=True)
class AblationPublication:
    variant_id: str
    run_id: str
    kernel_slug: str
    checkpoint_sha256: str
    expected_final_metrics: tuple[tuple[str, float], ...]
    config_items: tuple[tuple[str, object], ...]
    operational_definition: str
    source_commit: str = SOURCE_COMMIT
    baseline_run_id: str = BASELINE_RUN_ID
    baseline_label: str = BASELINE_LABEL
    dataset: str = "MKG-W"
    kaggle_version: int = 1
    epochs: int = 2000
    eval_freq: int = 100
    metric_policy: str = METRIC_POLICY

    @property
    def config(self) -> dict[str, object]:
        """Return a fresh JSON-serializable view of the frozen config."""

        return dict(self.config_items)

    @property
    def final_metrics(self) -> dict[str, float]:
        return dict(self.expected_final_metrics)


def _config(*, mu: float, use_exid: bool, exid_mode: str, use_gating_noise: bool,
            use_relation_temperature: bool) -> tuple[tuple[str, object], ...]:
    return tuple(
        {
            "lr": 0.001,
            "mu": mu,
            "dim": 200,
            "r_dim": 256,
            "dataset": "MKG-W",
            "epochs": 2000,
            "batch_size": 1024,
            "n_exp": 3,
            "seed": 10010,
            "eval_freq": 100,
            "save": 1,
            "use_exid": use_exid,
            "exid_mode": exid_mode,
            "use_gating_noise": use_gating_noise,
            "use_relation_temperature": use_relation_temperature,
        }.items()
    )


PUBLICATION_VARIANTS: Mapping[str, AblationPublication] = {
    "no_gating_noise": AblationPublication(
        variant_id="no_gating_noise",
        run_id="mkgwngn1",
        kernel_slug="momok-mkg-w-no-gating-noise-full",
        checkpoint_sha256="58052dd66e83e61a58c1cc9bc5728dcb8285630280c23c77a89bed4e5caa3291",
        expected_final_metrics=(
            ("MRR", 0.3503), ("Hits@1", 0.2983), ("Hits@3", 0.3728), ("Hits@10", 0.4543)
        ),
        config_items=_config(
            mu=0.0001, use_exid=True, exid_mode="active", use_gating_noise=False,
            use_relation_temperature=True,
        ),
        operational_definition="Gaussian gating noise disabled in the existing MoE gating path; other model settings retained.",
    ),
    "no_exid": AblationPublication(
        variant_id="no_exid",
        run_id="mkgwnex1",
        kernel_slug="momok-mkg-w-no-exid-full",
        checkpoint_sha256="cf2d30c4883b1aec837da0aad6251ba5dd74b4aaf346d04307ad702cbaa81fc0",
        expected_final_metrics=(
            ("MRR", 0.3509), ("Hits@1", 0.2968), ("Hits@3", 0.3724), ("Hits@10", 0.4586)
        ),
        config_items=_config(
            mu=0.0, use_exid=False, exid_mode="disabled", use_gating_noise=True,
            use_relation_temperature=True,
        ),
        operational_definition="CLUB/ExID estimator and its optimizer disabled; mu=0.0; other model settings retained.",
    ),
    "no_relation_temperature": AblationPublication(
        variant_id="no_relation_temperature",
        run_id="mkgwnrt1",
        kernel_slug="momok-mkg-w-no-relation-temperature-full",
        checkpoint_sha256="6aa771d6c985fec4ae97225a77c1f24ed40631b6e8e493c8f0607755833d6898",
        expected_final_metrics=(
            ("MRR", 0.3586), ("Hits@1", 0.3059), ("Hits@3", 0.3759), ("Hits@10", 0.4628)
        ),
        config_items=_config(
            mu=0.0001, use_exid=True, exid_mode="active", use_gating_noise=True,
            use_relation_temperature=False,
        ),
        operational_definition="Relation-conditioned temperature removed from expert gating; relation embeddings used by prediction heads remain active.",
    ),
}


def publication_for(variant_id: str) -> AblationPublication:
    try:
        return PUBLICATION_VARIANTS[variant_id]
    except KeyError as exc:
        raise VariantPublicationError(f"unsupported publication variant: {variant_id}") from exc


def validate_dataset_manifest(manifest: dict, item: AblationPublication) -> None:
    if not isinstance(manifest, dict):
        raise VariantPublicationError("dataset manifest must be a JSON object")
    for key, expected in DATASET_FINGERPRINT.items():
        actual = manifest.get(key)
        if isinstance(expected, tuple):
            expected = list(expected)
        if actual != expected:
            raise VariantPublicationError(
                f"{item.variant_id}: dataset manifest {key}={actual!r}, expected {expected!r}"
            )


def safe_output_files() -> tuple[str, ...]:
    """Return the exact allowlist for Kaggle output recovery."""

    return SAFE_OUTPUT_FILES


def validate_safe_output_dir(output_dir, item: AblationPublication) -> None:
    """Validate a flattened allowlisted download without opening checkpoints."""

    from pathlib import Path
    import json

    root = Path(output_dir)
    actual_names = {path.name for path in root.iterdir() if path.is_file()}
    expected_names = set(SAFE_OUTPUT_FILES)
    if actual_names != expected_names:
        raise VariantPublicationError(
            f"{item.variant_id}: recovered files {sorted(actual_names)!r}, "
            f"expected exactly {sorted(expected_names)!r}"
        )
    if (root / "checkpoint_final.pt").exists():
        raise VariantPublicationError("checkpoint binaries are not part of the safe download")
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    validate_dataset_manifest(manifest, item)
    inventory = json.loads((root / "output_inventory.json").read_text(encoding="utf-8"))
    if not isinstance(inventory, list) or "checkpoints/checkpoint_final.pt" not in inventory:
        raise VariantPublicationError("output inventory does not identify checkpoint_final.pt")
    sidecar = (root / "checkpoint_final.pt.sha256").read_text(encoding="utf-8").strip()
    digest = sidecar.split()[0] if sidecar else ""
    if digest != item.checkpoint_sha256:
        raise VariantPublicationError(
            f"{item.variant_id}: checkpoint sidecar SHA256 does not match publication record"
        )
    if not (root / "ablation_smoke.log").stat().st_size:
        raise VariantPublicationError("ablation training log is empty")


def comparison_rows(items: Sequence[AblationPublication]) -> list[list[object]]:
    """Rows for an exact baseline/ablation comparison, preserving full precision."""

    rows: list[list[object]] = []
    for metric, baseline in BASELINE_METRICS.items():
        rows.append([BASELINE_LABEL, BASELINE_RUN_ID, metric, baseline, baseline, 0.0, 0.0, 0.0])
    for item in items:
        for metric, reproduced in item.expected_final_metrics:
            baseline = BASELINE_METRICS[metric]
            signed = reproduced - baseline
            absolute = abs(signed)
            rows.append(
                [
                    item.variant_id,
                    item.run_id,
                    metric,
                    reproduced,
                    baseline,
                    signed,
                    absolute,
                    absolute / baseline * 100.0,
                ]
            )
    return rows
