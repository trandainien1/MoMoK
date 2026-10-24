"""Process launcher boundary for the ablation runtime.

The command builder is kept separate from the Torch-dependent process so it
can be verified on the local host before any Kaggle execution is authorized.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .artifact_bundle import create_resume_bundle
from .logging_contract import RunLayout
from .variants import config_fingerprint


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    layout: RunLayout


def build_train_command(
    args,
    *,
    train_script: str | Path,
    output_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(train_script),
        "--cuda",
        "0",
        "--dataset",
        args.dataset,
        "--variant",
        args.variant,
        "--lr",
        str(args.lr),
        "--mu",
        str(args.mu),
        "--dim",
        str(args.dim),
        "--r_dim",
        str(args.r_dim),
        "--batch_size",
        str(args.batch_size),
        "--n_exp",
        str(args.n_exp),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--eval_freq",
        str(args.eval_freq),
        "--checkpoint_every",
        str(args.checkpoint_every),
        "--max_train_batches",
        str(args.max_train_batches or 0),
        "--output_dir",
        str(output_dir if output_dir is not None else args.output_dir),
        "--run_mode",
        args.run_mode,
        "--metric_policy",
        args.metric_policy,
        "--wandb_mode",
        args.wandb_mode,
    ]
    if args.resume_from is not None:
        command.extend(["--resume_checkpoint", str(args.resume_from)])
    if checkpoint_dir is not None:
        command.extend(["--checkpoint_dir", str(checkpoint_dir)])
    return command


def run_process(args, *, train_script: str | Path, base_output: Path) -> ProcessResult:
    """Run one controlled process and persist its configuration/log output."""

    config = {
        key: value
        for key, value in vars(args).items()
        if key != "variant_config"
    }
    config["variant_config"] = args.variant_config.to_dict()
    config = json.loads(json.dumps(config, default=str))
    fingerprint = config_fingerprint(config)
    layout = RunLayout.create(
        base_output,
        dataset=args.dataset,
        variant=args.variant,
        seed=args.seed,
        config_hash=fingerprint,
    )
    (layout.root / "config.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "variant": args.variant,
                "seed": args.seed,
                "config_hash": fingerprint,
                "config": config,
            },
            indent=2,
            sort_keys=True,
        )
    )

    command = build_train_command(
        args,
        train_script=train_script,
        output_dir=layout.root,
        checkpoint_dir=layout.checkpoints,
    )
    completed = subprocess.run(
        command,
        cwd=Path(train_script).resolve().parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (layout.logs / "session_001.log").write_text(completed.stdout)
    checkpoint = layout.checkpoints / "checkpoint_latest.pt"
    if checkpoint.is_file():
        create_resume_bundle(layout)
    if completed.returncode != 0:
        (layout.reports / "failure.json").write_text(
            json.dumps(
                {
                    "phase": "training_process",
                    "last_completed_epoch": 0,
                    "exit_status": "failed",
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                },
                indent=2,
            )
        )
    return ProcessResult(completed.returncode, completed.stdout, layout)
