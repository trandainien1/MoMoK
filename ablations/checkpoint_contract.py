"""Framework-independent checkpoint schema for ablation runs."""

from typing import Iterable


BASE_STATE_KEYS = (
    "format_version",
    "completed_epoch",
    "model_state_dict",
    "optimizer_state_dict",
    "lr_scheduler_state_dict",
    "best_validation_metrics",
    "python_random_state",
    "numpy_rng_state",
    "torch_rng_state",
    "torch_cuda_rng_state_all",
    "corpus_train_indices",
    "config",
    "config_hash",
    "variant_id",
    "dataset",
    "source_commit",
)
ESTIMATOR_STATE_KEYS = ("estimator_state_dict", "optimizer_mi_state_dict")


def required_state_keys(*, use_exid: bool) -> tuple[str, ...]:
    """Return required checkpoint keys for a variant's estimator mode."""

    keys = list(BASE_STATE_KEYS)
    if use_exid:
        keys.extend(ESTIMATOR_STATE_KEYS)
    return tuple(keys)


def build_checkpoint_metadata(
    *,
    variant_id: str,
    config_hash: str,
    dataset: str,
    source_commit: str,
    completed_epoch: int,
    n_exp: int,
) -> dict:
    """Build the non-tensor metadata shared by save and resume validation."""

    if completed_epoch < 0:
        raise ValueError("completed_epoch must be non-negative")
    if n_exp < 1:
        raise ValueError("n_exp must be positive")
    return {
        "format_version": 1,
        "completed_epoch": int(completed_epoch),
        "variant_id": variant_id,
        "config_hash": config_hash,
        "dataset": dataset,
        "source_commit": source_commit,
        "n_exp": int(n_exp),
    }


def missing_required_keys(payload: dict, *, use_exid: bool) -> tuple[str, ...]:
    """Return missing schema keys without loading tensor values."""

    required = required_state_keys(use_exid=use_exid)
    return tuple(key for key in required if key not in payload)
