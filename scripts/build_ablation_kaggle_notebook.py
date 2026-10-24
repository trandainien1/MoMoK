#!/usr/bin/env python3
"""Build a bounded, self-contained MoMoK ablation Kaggle notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "99c2df114d48c79708ea0608644b685183d81bd9"
DRIVE_ID = "1dKJdJunb11kDtFr5NLfPlFknS7cRdm9W"
SOURCE_FILES = ("train.py", "models/RelMoE.py", "models/modules.py", "models/model.py", "utils/data_loader.py", "utils/data_util.py", "layers/layer.py", "ablations/__init__.py", "ablations/variants.py", "ablations/runtime_contract.py", "ablations/checkpoint_contract.py", "ablations/checkpoints.py", "ablations/torch_checkpoint.py")
EXPECTED = {"MKG-W": {"entities": 15000, "relations": 169, "train": 34196, "valid": 4276, "test": 4274}, "MKG-Y": {"entities": 15000, "relations": 28, "train": 21310, "valid": 2665, "test": 2663}}


def cell(kind: str, source: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def build_notebook(*, output: Path, dataset: str, variant: str, lr: float, mu: float, dim: int, n_exp: int, epochs: int, run_mode: str) -> None:
    if dataset not in EXPECTED:
        raise ValueError(f"unsupported dataset: {dataset}")
    overlay = {name: (ROOT / name).read_text() for name in SOURCE_FILES}
    expected = EXPECTED[dataset]
    batches = 2 if run_mode == "smoke" else 0
    command = ["python", "-u", "train.py", "--cuda", "0", "--dataset", dataset, "--variant", variant, "--lr", str(lr), "--mu", str(mu), "--dim", str(dim), "--batch_size", "1024", "--n_exp", str(n_exp), "--seed", "10010", "--epochs", str(epochs), "--eval_freq", "1" if run_mode == "smoke" else "100", "--save", "1", "--checkpoint_every", "1" if run_mode == "smoke" else "25", "--checkpoint_dir", "/kaggle/working/momok_ablation/checkpoints", "--output_dir", "/kaggle/working/momok_ablation", "--run_mode", run_mode, "--max_train_batches", str(batches)]
    config = f"from pathlib import Path\nimport json, os, shutil, subprocess, sys, zipfile\nREPO_URL={('https://github.com/zjukg/MoMoK.git')!r}\nSOURCE_COMMIT={SOURCE_COMMIT!r}\nDATASET={dataset!r}\nVARIANT={variant!r}\nEXPECTED={expected!r}\nOUTPUT_DIR=Path('/kaggle/working/momok_ablation')\nREPO_DIR=Path('/kaggle/working/MoMoK')\nOUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    clone = "subprocess.check_call(['git','clone',REPO_URL,str(REPO_DIR)])\nsubprocess.check_call(['git','-C',str(REPO_DIR),'checkout','--detach',SOURCE_COMMIT])\nos.environ['MOMOK_SOURCE_COMMIT']=SOURCE_COMMIT\nprint('SOURCE_PIN=PASS')\n"
    install = "subprocess.check_call([sys.executable,'-m','pip','install','-q','gdown','tqdm'])\nimport torch\nassert torch.cuda.is_available()\nprint('GPU=PASS')\nprint('EFFECTIVE_GPU='+torch.cuda.get_device_name(0))\n"
    features = f"archive=Path('/kaggle/working/embeddings.zip')\nsubprocess.check_call([sys.executable,'-m','gdown','--id',{DRIVE_ID!r},'-O',str(archive)])\nextract=Path('/kaggle/working/embeddings_extracted')\nextract.mkdir(exist_ok=True)\nwith zipfile.ZipFile(archive) as z:\n    names=z.namelist()\n    for member in (f'embeddings/{{DATASET}}/img_features.pth',f'embeddings/{{DATASET}}/text_features.pth'):\n        assert names.count(member)==1\n        z.extract(member,extract)\ndataset_dir=REPO_DIR/'datasets'/DATASET\ndataset_dir.mkdir(parents=True,exist_ok=True)\nfor name in ('img_features.pth','text_features.pth'):\n    shutil.copy2(extract/'embeddings'/DATASET/name,dataset_dir/name)\nprint('SELECTED_IMG_MEMBER=embeddings/'+DATASET+'/img_features.pth')\nprint('SELECTED_TEXT_MEMBER=embeddings/'+DATASET+'/text_features.pth')\n"
    validate = "required=['entity2id.txt','relation2id.txt','train.txt','valid.txt','test.txt','img_features.pth','text_features.pth']\nassert not [n for n in required if not (dataset_dir/n).is_file()]\ncounts={k:sum(1 for _ in (dataset_dir/f).open()) for k,f in {'entities':'entity2id.txt','relations':'relation2id.txt','train':'train.txt','valid':'valid.txt','test':'test.txt'}.items()}\nimg=torch.load(dataset_dir/'img_features.pth',map_location='cpu',weights_only=False)\ntxt=torch.load(dataset_dir/'text_features.pth',map_location='cpu',weights_only=False)\nassert counts==EXPECTED and list(img.shape)==[EXPECTED['entities'],383] and list(txt.shape)==[EXPECTED['entities'],384]\nassert torch.isfinite(img).all() and torch.isfinite(txt).all()\n(OUTPUT_DIR/'dataset_manifest.json').write_text(json.dumps({**counts,'image_shape':list(img.shape),'text_shape':list(txt.shape)},indent=2))\nprint('DATASET_VALIDATION=PASS')\n"
    overlay_cell = f"OVERLAY={overlay!r}\nfor relative,source in OVERLAY.items():\n    target=REPO_DIR/relative\n    target.parent.mkdir(parents=True,exist_ok=True)\n    target.write_text(source)\nprint('ABLATION_SOURCE_OVERLAY=PASS')\n"
    run = f"command={command!r}\nenv=os.environ.copy()\nenv['PYTHONPATH']=str(REPO_DIR)\nlog=OUTPUT_DIR/'ablation_smoke.log'\nwith log.open('w') as handle:\n    result=subprocess.run(command,cwd=REPO_DIR,env=env,text=True,stdout=handle,stderr=subprocess.STDOUT)\nprint('TRAIN_EXIT_CODE='+str(result.returncode))\nif result.returncode: raise RuntimeError('ablation run failed; inspect '+str(log))\nprint('RUN_MODE_COMPLETE={run_mode}')\n"
    inventory = "inventory=sorted(str(p.relative_to(OUTPUT_DIR)) for p in OUTPUT_DIR.rglob('*') if p.is_file())\n(OUTPUT_DIR/'output_inventory.json').write_text(json.dumps(inventory,indent=2))\nprint('OUTPUT_INVENTORY=PASS')\n"
    notebook = {"cells": [cell("markdown", f"# MoMoK Ablation\nDataset: `{dataset}`\nVariant: `{variant}`\nPinned source: `{SOURCE_COMMIT}`\n"), cell("code", config), cell("code", clone), cell("code", overlay_cell), cell("code", install), cell("code", features), cell("code", validate), cell("code", run), cell("code", inventory)], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "kaggle": {"accelerator": "NvidiaTeslaT4", "internet": True, "is_private": True}}, "nbformat": 4, "nbformat_minor": 5}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "kaggle/momok-ablation/momok_ablation_reproduction.ipynb")
    parser.add_argument("--dataset", choices=sorted(EXPECTED), default="MKG-W")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--mu", type=float, default=0.0001)
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--n_exp", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--run_mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    build_notebook(output=args.output, dataset=args.dataset, variant=args.variant, lr=args.lr, mu=args.mu, dim=args.dim, n_exp=args.n_exp, epochs=args.epochs, run_mode=args.run_mode)
    print("ABLATION_NOTEBOOK_BUILD=PASS")


if __name__ == "__main__":
    main()
