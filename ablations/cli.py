"""Argument parsing for the variant runner.

The parser is intentionally independent from the legacy ``train.py``
module, whose top-level import starts the upstream runtime immediately.
"""

import argparse
from pathlib import Path

from .variants import VARIANT_IDS, VariantConfigError, resolve_variant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MoMoK ablation runner")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANT_IDS))
    parser.add_argument("--n_exp", type=int, default=3)
    parser.add_argument("--mu", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=10010)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--r_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resume_from", type=Path, default=None)
    parser.add_argument("--run_mode", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--metric_policy",
        choices=("legacy", "validation_selected"),
        default="validation_selected",
    )
    parser.add_argument(
        "--wandb_mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
    )
    return parser


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.variant_config = resolve_variant(
            args.variant,
            n_exp=args.n_exp,
            mu=args.mu,
        )
    except VariantConfigError as exc:
        raise ValueError(str(exc)) from exc

    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.eval_freq < 1 or args.checkpoint_every < 1:
        raise ValueError("eval_freq and checkpoint_every must be positive")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise ValueError("max_train_batches must be positive when supplied")
    if args.run_mode == "smoke" and args.max_train_batches is None:
        raise ValueError("smoke mode requires --max_train_batches")
    if args.resume_from is not None and not args.resume_from:
        raise ValueError("resume_from cannot be empty")
    return args
