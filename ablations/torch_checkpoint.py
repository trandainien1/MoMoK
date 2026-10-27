"""Torch-backed full-state checkpoint helpers.

PyTorch is imported lazily so configuration and schema tests remain runnable
on the local CPU-only host.
"""

import copy
import io
import os
import random
from pathlib import Path

import numpy as np

from .checkpoints import atomic_save, validate_payload
from .variants import config_fingerprint


def checkpoint_config(args) -> dict:
    variant = getattr(args, "variant_config", None)
    return {
        "dataset": args.dataset,
        "variant": args.variant,
        "seed": args.seed,
        "lr": args.lr,
        "mu": args.mu,
        "dim": args.dim,
        "r_dim": args.r_dim,
        "batch_size": args.batch_size,
        "n_exp": args.n_exp,
        "epochs": args.epochs,
        "eval_freq": args.eval_freq,
        "metric_policy": getattr(args, "metric_policy", "legacy"),
        "variant_config": variant.to_dict() if variant is not None else {},
    }


def build_payload(
    args,
    model,
    estimator,
    optimizer,
    optimizer_mi,
    lr_scheduler,
    best_validation_metrics,
    corpus,
    completed_epoch: int,
    final_epoch_metrics=None,
) -> dict:
    import torch

    config = checkpoint_config(args)
    payload = {
        "format_version": 1,
        "completed_epoch": int(completed_epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "best_validation_metrics": copy.deepcopy(best_validation_metrics),
        "final_epoch_metrics": copy.deepcopy(final_epoch_metrics),
        "python_random_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
        "corpus_train_indices": copy.deepcopy(corpus.train_indices),
        "config": config,
        "config_hash": config_fingerprint(config),
        "variant_id": args.variant,
        "dataset": args.dataset,
        "source_commit": os.environ.get("MOMOK_SOURCE_COMMIT", ""),
        "n_exp": args.n_exp,
    }
    if estimator is not None and optimizer_mi is not None:
        payload["estimator_state_dict"] = estimator.state_dict()
        payload["optimizer_mi_state_dict"] = optimizer_mi.state_dict()
    return payload


def save_checkpoint(path: Path, payload: dict) -> str:
    import torch

    def serialize(value):
        buffer = io.BytesIO()
        torch.save(value, buffer)
        return buffer.getvalue()

    return atomic_save(payload, Path(path), serializer=serialize)


def _move_optimizer_state(optimizer, device) -> None:
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for key, value in state.items():
            if hasattr(value, "to"):
                state[key] = value.to(device)


def restore_checkpoint(
    path: Path,
    *,
    args,
    model,
    estimator,
    optimizer,
    optimizer_mi,
    lr_scheduler,
    corpus,
) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint_config(args)
    validate_payload(
        payload,
        expected_config=config,
        use_exid=estimator is not None,
        allow_epoch_extension=True,
    )
    if payload.get("source_commit", "") != os.environ.get("MOMOK_SOURCE_COMMIT", ""):
        raise ValueError("checkpoint source commit mismatch")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    lr_scheduler.load_state_dict(payload["lr_scheduler_state_dict"])
    if estimator is not None:
        estimator.load_state_dict(payload["estimator_state_dict"])
        optimizer_mi.load_state_dict(payload["optimizer_mi_state_dict"])
    _move_optimizer_state(optimizer, args.device)
    _move_optimizer_state(optimizer_mi, args.device)
    corpus.train_indices = copy.deepcopy(payload["corpus_train_indices"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state_all"])
    return payload
