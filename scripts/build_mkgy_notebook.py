#!/usr/bin/env python3
"""Build the separate MKG-Y notebook from the audited hardened template."""

from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "repro" / "configs" / "mkgy_main.json"
WANDB_RUN_ID_PATH = ROOT / "repro" / "state" / "mkgy_wandb_run_id.txt"
TEMPLATE = ROOT / "kaggle" / "momok-db15k" / "momok_db15k_reproduction.ipynb"
OUT_DIR = ROOT / "kaggle" / "momok-mkg-y"
OUT = OUT_DIR / "momok_mkgy_reproduction.ipynb"


def replace_config_literal(source: str, config: dict) -> str:
    fingerprint = config["dataset_fingerprint"]
    metrics = config["paper_metrics"]
    source = source.replace("MKG-W", "MKG-Y").replace("mkgw", "mkgy").replace("MKG_W", "MKG_Y")
    source = source.replace("0.3589", "0.3791").replace("0.3038", "0.3509").replace("0.3754", "0.3920").replace("0.4613", "0.4320")
    source = source.replace("34196", str(fingerprint["train"])).replace("4276", str(fingerprint["valid"])).replace("4274", str(fingerprint["test"])).replace("169", str(fingerprint["relations"]))
    source = source.replace("14463", str(fingerprint["image_num"])).replace("14123", str(fingerprint["text_num"]))
    source = source.replace("LR = 0.001", "LR = 0.0005").replace("DIM = 200", "DIM = 300")
    source = source.replace("'lr': 0.001", "'lr': 0.0005").replace("'dim': 200", "'dim': 300")
    source = source.replace('CONFIG_SOURCE = "OFFICIAL_README"', 'CONFIG_SOURCE = "OFFICIAL_README_PLUS_PINNED_DEFAULTS"')
    source = source.replace('ACTIVE_KERNEL_REF = "nientrandai1/momok-mkg-w-reproduction"', 'ACTIVE_KERNEL_REF = "nientrandai1/momok-mkg-y-reproduction"')
    source = source.replace('RESUME_DATASET_REF = "nientrandai1/momok-mkgw-resume-state"', 'RESUME_DATASET_REF = "nientrandai1/momok-mkgy-resume-state"')
    source = source.replace('PAPER_METRICS = {\'MRR\': 0.3589, \'Hits@1\': 0.3038, \'Hits@3\': 0.3754, \'Hits@10\': 0.4613}', f'PAPER_METRICS = {metrics!r}')
    source = source.replace('EXPECTED_DATASET = {\'entities\': 15000, \'relations\': 169, \'train\': 34196, \'valid\': 4276, \'test\': 4274, \'image_available\': 14463, \'image_dim\': 383, \'text_available\': 14123, \'text_dim\': 384}', f'EXPECTED_DATASET = {{\'entities\': {fingerprint["entities"]}, \'relations\': {fingerprint["relations"]}, \'train\': {fingerprint["train"]}, \'valid\': {fingerprint["valid"]}, \'test\': {fingerprint["test"]}, \'image_available\': {fingerprint["image_num"]}, \'image_dim\': {fingerprint["image_dim"]}, \'text_available\': {fingerprint["text_num"]}, \'text_dim\': {fingerprint["text_dim"]}}}')
    source = source.replace("RESUME_MODE = \"FRESH_RESTART_WITH_CHECKPOINTING\"", 'RESUME_MODE = "auto"')
    source = source.replace("RESUME_FROM_DB15K=NO", "RESUME_FROM_MKG_W=NO")
    source = source.replace("NO_ACTIVE_DB15K_KERNEL_REF=YES", "NO_ACTIVE_MKG_W_KERNEL_REF=YES")
    source = source.replace("DATASET = \"MKG-Y\"\nCONFIG_SOURCE", 'DATASET = "MKG-Y"\nWANDB_ENTITY = "trandainien1"\nWANDB_PROJECT = "graphml-momok-reproduction"\nCONFIG_SOURCE')
    return source


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    wandb_run_id = WANDB_RUN_ID_PATH.read_text().strip()
    if not wandb_run_id or any(ch.isspace() for ch in wandb_run_id):
        raise RuntimeError("stable MKG-Y W&B run ID is missing or invalid")
    nb = nbformat.read(TEMPLATE, as_version=4)
    for cell in nb.cells:
        if "source" in cell:
            cell.source = replace_config_literal(cell.source, config)

    by_prefix = {cell.source.splitlines()[0] if cell.source.splitlines() else "": cell for cell in nb.cells if cell.cell_type == "code"}
    config_cell = next(cell for cell in nb.cells if 'DATASET = "MKG-Y"' in cell.source and 'WANDB_ENTITY' in cell.source)
    config_cell.source = config_cell.source.replace('print("ACTIVE_KERNEL_REF=" + ACTIVE_KERNEL_REF)', 'print("ACTIVE_KERNEL_REF=" + ACTIVE_KERNEL_REF)\nprint("CONFIG_SOURCE=" + CONFIG_SOURCE)')
    config_cell.source = config_cell.source.replace(
        'WANDB_PROJECT = "graphml-momok-reproduction"',
        'WANDB_PROJECT = "graphml-momok-reproduction"\nWANDB_RUN_ID = ' + repr(wandb_run_id) +
        '\nprint("WANDB_STABLE_RUN_ID=YES")',
    )

    manifest_cell = next(cell for cell in nb.cells if 'dataset_dir = Path(REPO_DIR, "datasets", DATASET)' in cell.source)
    manifest_cell.source = manifest_cell.source.replace('print("FEATURE_CONTRACT=PASS")', 'print("FEATURE_CONTRACT=PASS")\nprint("MKG_Y_DATASET_VALIDATION=PASS")')

    wandb_cell = next(cell for cell in nb.cells if 'WANDB_STATUS = "NOT_CONFIGURED"' in cell.source)
    wandb_cell.source = wandb_cell.source.replace('WANDB_STATUS = "NOT_CONFIGURED"', 'WANDB_STATUS = "NOT_CONFIGURED"\nWANDB_MODE = "disabled"')
    wandb_cell.source = wandb_cell.source.replace('secret = UserSecretsClient().get_secret("WANDB_API_KEY")', 'secret = UserSecretsClient().get_secret("WANDB_API_KEY")\n    os.environ["WANDB_ENTITY"] = WANDB_ENTITY\n    os.environ["WANDB_PROJECT"] = WANDB_PROJECT')
    wandb_cell.source = wandb_cell.source.replace('WANDB_STATUS = "CONFIGURED"', 'WANDB_STATUS = "CONFIGURED"\n        WANDB_MODE = "online"')
    wandb_cell.source += '\nif not WANDB_ENABLED:\n    raise RuntimeError("HUMAN_GATE_REQUIRED: Add WANDB_API_KEY to Kaggle Secrets before full training")\nprint("WANDB_LIVE=YES")\n'

    live_cell = next(cell for cell in nb.cells if 'WANDB_RUN = wandb.init' in cell.source)
    live_cell.source = live_cell.source.replace(
        'WANDB_RUN = wandb.init(\n',
        'WANDB_RUN = wandb.init(\n        id=WANDB_RUN_ID,\n        resume="allow",\n',
    )
    live_cell.source = live_cell.source.replace('project=os.environ.get("WANDB_PROJECT", "graphml-momok-reproduction"),', 'name="momok-mkgy-main-kaggle-v1-seed10010",\n        tags=["MoMoK", "ICLR-2025", "MKG-Y", "main-result", "reproduction", "Kaggle", "2000-epochs", "live-tracking"],\n        project=os.environ.get("WANDB_PROJECT", "graphml-momok-reproduction"),')
    live_cell.source = live_cell.source.replace('config={"dataset": DATASET, "seed": SEED, "epochs": EPOCHS, "tracking_mode": "live_kaggle"},', 'config={"paper_title": "MoMoK", "paper_venue": "ICLR 2025", "dataset": DATASET, "seed": SEED, "lr": LR, "mu": MU, "dim": DIM, "batch_size": BATCH_SIZE, "n_exp": N_EXPERTS, "epochs": EPOCHS, "eval_freq": EVAL_FREQ, "source_commit": REPO_COMMIT, "kaggle_kernel": ACTIVE_KERNEL_REF, "wandb_run_id": WANDB_RUN_ID, "tracking_mode": "live_kaggle", "paper_target_mrr": PAPER_METRICS["MRR"], "paper_target_hits1": PAPER_METRICS["Hits@1"], "paper_target_hits3": PAPER_METRICS["Hits@3"], "paper_target_hits10": PAPER_METRICS["Hits@10"]},')
    live_cell.source = live_cell.source.replace('reinit="finish_previous",', 'reinit="finish_previous",\n        settings=wandb.Settings(start_method="thread"),')
    live_cell.source = live_cell.source.replace(
        '    epoch_match = re.search',
        '''    train_marker = re.search(r"TRAIN_EPOCH_METRICS_JSON=(\\{.*\\})", line)
    if train_marker:
        payload = json.loads(train_marker.group(1))
        _WANDB_EPOCH = int(payload["epoch"])
        WANDB_RUN.log({"epoch": _WANDB_EPOCH, "train/loss": float(payload["loss"]), "runtime/epoch_time_sec": float(payload["epoch_time_sec"])})
    epoch_match = re.search''',
    )
    live_cell.source = live_cell.source.replace('import os, re', 'import json, os, re')

    for cell in nb.cells:
        if cell.cell_type == "code" and "_checkpoint_payload(args" in cell.source:
            cell.source = cell.source.replace(
                "'format_version': 2,\n        'completed_epoch': int(completed_epoch),",
                "'format_version': 2,\n        'completed_epoch': int(completed_epoch),\n        'wandb_run_id': os.environ.get('MOMOK_WANDB_RUN_ID', ''),",
            )
            cell.source = cell.source.replace(
                "        final_epoch_metrics = None\n        if completed_epoch % args.eval_freq == 0:",
                "        final_epoch_metrics = None\n        print('TRAIN_EPOCH_METRICS_JSON=' + json.dumps({'epoch': completed_epoch, 'loss': float(sum(epoch_loss) / len(epoch_loss)), 'epoch_time_sec': float(time.time() - t)}))\n        if completed_epoch % args.eval_freq == 0:",
            )
            cell.source = cell.source.replace(
                "'training_config': {",
                "'training_config': {",
            )
            cell.source = cell.source.replace(
                "'training_config', 'source_commit', 'dataset_metadata',",
                "'training_config', 'source_commit', 'dataset_metadata', 'wandb_run_id',",
            )
            cell.source = cell.source.replace(
                "        if checkpoint['source_commit'] != os.environ.get('MOMOK_SOURCE_COMMIT', ''):",
                "        if checkpoint.get('wandb_run_id', '') != os.environ.get('MOMOK_WANDB_RUN_ID', ''):\n            raise RuntimeError('Checkpoint W&B run ID mismatch')\n        if checkpoint['source_commit'] != os.environ.get('MOMOK_SOURCE_COMMIT', ''):",
            )
            break

    for cell in nb.cells:
        if cell.cell_type == "code" and 'run_env["MOMOK_SOURCE_COMMIT"]' in cell.source:
            cell.source = cell.source.replace(
                'run_env["MOMOK_SOURCE_COMMIT"] = REPO_COMMIT',
                'run_env["MOMOK_SOURCE_COMMIT"] = REPO_COMMIT\nrun_env["MOMOK_WANDB_RUN_ID"] = WANDB_RUN_ID',
            )

    inventory_cell = next(cell for cell in nb.cells if 'OUTPUT_INVENTORY=PASS' in cell.source)
    inventory_cell.source = inventory_cell.source.replace(
        'for path in sorted(Path(OUTPUT_DIR).rglob("*")):',
        '''if not globals().get("PARTIAL_RUN", False) and WANDB_RUN is not None:
    summary = json.loads(Path(OUTPUT_DIR, "repro_summary.json").read_text())
    for key, value in {
        "completed_epoch": 2000,
        "paper/MRR": summary["paper_targets"]["MRR"],
        "paper/Hits@1": summary["paper_targets"]["Hits@1"],
        "paper/Hits@3": summary["paper_targets"]["Hits@3"],
        "paper/Hits@10": summary["paper_targets"]["Hits@10"],
        "reproduced/MRR": summary["reproduced"]["MRR"],
        "reproduced/Hits@1": summary["reproduced"]["Hits@1"],
        "reproduced/Hits@3": summary["reproduced"]["Hits@3"],
        "reproduced/Hits@10": summary["reproduced"]["Hits@10"],
        "reproduction_verdict": summary["verdict"],
        "dataset_identity_verified": True,
        "tracking_mode": "live_kaggle",
    }.items():
        WANDB_RUN.summary[key] = value
    comparison = wandb.Table(columns=["Metric", "Paper", "Reproduced", "Absolute Delta", "Relative Delta (%)"], data=[
        [metric, summary["paper_targets"][metric], summary["reproduced"][metric], abs(summary["absolute_delta"][metric]), abs(summary["relative_delta"][metric]) * 100]
        for metric in ["MRR", "Hits@1", "Hits@3", "Hits@10"]
    ])
    WANDB_RUN.log({"paper_vs_reproduction": comparison})
    environment_rows = [
        ["Dataset", DATASET], ["Entities", EXPECTED_DATASET["entities"]], ["Relations", EXPECTED_DATASET["relations"]],
        ["Train", EXPECTED_DATASET["train"]], ["Valid", EXPECTED_DATASET["valid"]], ["Test", EXPECTED_DATASET["test"]],
        ["Epochs", EPOCHS], ["Seed", SEED], ["LR", LR], ["Mu", MU], ["Dim", DIM], ["Batch size", BATCH_SIZE], ["Experts", N_EXPERTS],
        ["GPU", environment["gpu"]], ["Python", environment["effective_python"]], ["Torch", environment["effective_torch"]], ["CUDA", environment["effective_cuda"]],
        ["Kaggle kernel", ACTIVE_KERNEL_REF], ["Source commit", REPO_COMMIT],
    ]
    environment_table = wandb.Table(columns=["Key", "Value"], data=[[str(key), str(value)] for key, value in environment_rows])
    WANDB_RUN.log({"reproduction_environment": environment_table})
    Path(OUTPUT_DIR, "provenance.json").write_text(json.dumps({"tracking_mode": "live_kaggle", "wandb_run_id": WANDB_RUN_ID, "dataset": DATASET, "source_commit": REPO_COMMIT, "kaggle_kernel": ACTIVE_KERNEL_REF}, indent=2))
    evidence = wandb.Artifact("momok-mkg-y-kaggle-evidence", type="reproduction-evidence", metadata={
        "dataset": DATASET, "source_commit": REPO_COMMIT, "completed_epoch": 2000,
        "verdict": summary["verdict"], "wandb_run_id": WANDB_RUN_ID,
    })
    for name in ["repro_summary.json", "REPRODUCTION_REPORT.md", "dataset_manifest.json", "environment.json", "source.json", "source_fixes.json", "upstream_compatibility.patch", "smoke_summary.json", "provenance.json"]:
        path = Path(OUTPUT_DIR, name)
        if path.exists(): evidence.add_file(str(path), name=name)
    checkpoint_hash = Path(OUTPUT_DIR, "checkpoints", "checkpoint_final.pt.sha256")
    if checkpoint_hash.exists(): evidence.metadata["checkpoint_sha256_record"] = checkpoint_hash.read_text().strip()
    WANDB_RUN.log_artifact(evidence)
    WANDB_RUN.finish()

for path in sorted(Path(OUTPUT_DIR).rglob("*")):''',
    )

    full_cell = next(cell for cell in nb.cells if 'FULL_COMMAND=' in cell.source)
    full_cell.source = full_cell.source.replace('log_path = Path(OUTPUT_DIR, "mkgy_full.log")', 'log_path = Path(OUTPUT_DIR, "mkgy_full.log")')
    full_cell.source = full_cell.source.replace('if return_code != 0:', 'if return_code != 0:')

    nb.metadata.setdefault("kaggle", {})["accelerator"] = config["kernel"]["accelerator"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUT)
    print("MKGY_NOTEBOOK_BUILD=PASS")
    print(f"NOTEBOOK={OUT}")
    print(f"CELL_COUNT={len(nb.cells)}")


if __name__ == "__main__":
    main()
