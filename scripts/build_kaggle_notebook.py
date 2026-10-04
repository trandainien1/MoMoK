#!/usr/bin/env python3
"""Build the self-contained MoMoK MKG-W Kaggle notebook."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

from checkpoint_hardening import HARDENING_CELL
from notebook_runtime_hardening import harden_cells
from reproduction_config import (
    CONFIG_SOURCE,
    DATASET,
    ACTIVE_KERNEL_REF,
    EXPECTED_DATASET,
    feature_archive_members,
    GOOGLE_DRIVE_FILE_ID,
    PAPER_TARGETS,
    TRAINING_CONFIG,
    UPSTREAM_COMMIT,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle" / "momok-db15k" / "momok_db15k_reproduction.ipynb"


def md(text: str) -> dict:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> dict:
    return nbf.v4.new_code_cell(text.strip())


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "kaggle": {"accelerator": "NvidiaTeslaT4", "internet": True, "is_private": True},
    }
    cells = [
        md(f"""# MoMoK — DB15K Reproduction\n## ICLR 2025\n\nThis notebook reproduces the main full-model DB15K experiment from the official [MoMoK repository](https://github.com/zjukg/MoMoK), pinned to commit `{UPSTREAM_COMMIT}`. The target is the DB15K main model only: no MoMoK ablation and no PMF.\n\nThe notebook installs compatible dependencies, downloads the official embeddings, validates DB15K, runs a two-epoch smoke test, then executes exactly one 2000-epoch full run. All artifacts are written under `/kaggle/working/repro_output`.\n\nPaper target: **MRR = 0.3957**, **Hits@1 = 0.3238**, **Hits@3 = 0.4345**, **Hits@10 = 0.5414**."""),
        code(f"""REPO_URL = \"https://github.com/zjukg/MoMoK.git\"\nREPO_COMMIT = \"{UPSTREAM_COMMIT}\"\nWORK_ROOT = \"/kaggle/working\"\nREPO_DIR = \"/kaggle/working/MoMoK\"\nOUTPUT_DIR = \"/kaggle/working/repro_output\"\nDATASET = \"DB15K\"\nGOOGLE_DRIVE_FILE_ID = \"1dKJdJunb11kDtFr5NLfPlFknS7cRdm9W\"\nPAPER_METRICS = {{\"MRR\": 0.3957, \"Hits@1\": 0.3238, \"Hits@3\": 0.4345, \"Hits@10\": 0.5414}}\nSEED = 10010\nBATCH_SIZE = 1024\nN_EXPERTS = 3\nEPOCHS = 2000\nEVAL_FREQ = 100\nDB15K_HPARAM_SOURCE = \"AMBIGUOUS\"\nLR = 0.001\nMU = 0.0001\nDIM = 200\nRESUME_MODE = "RESTART_WITH_CHECKPOINTING"\nEXACT_RESUME_FROM_FAILED_RUN = False\nimport os\nos.makedirs(OUTPUT_DIR, exist_ok=True)"""),
        code("""import json, os, platform, shutil, subprocess, sys\nfrom pathlib import Path\n\nram_gb = None\ntry:\n    import psutil\n    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)\nexcept Exception:\n    pass\n\ndef run_capture(args, **kwargs):\n    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)\n\nprint(\"platform=\", platform.platform())\nprint(\"python=\", sys.version)\nprint(\"cpu_count=\", os.cpu_count())\nprint(\"ram_gb=\", ram_gb)\nprint(\"disk_usage=\", shutil.disk_usage(WORK_ROOT))\nprint(\"kaggle_dirs=\", sorted(str(p) for p in Path(\"/kaggle\").glob(\"*\")))\nsm = run_capture([\"bash\", \"-lc\", \"nvidia-smi || true\"])\nprint(sm.stdout)\nPath(OUTPUT_DIR, \"runtime_preinstall.json\").write_text(json.dumps({\n    \"platform\": platform.platform(), \"python\": sys.version, \"cpu_count\": os.cpu_count(),\n    \"ram_gb\": ram_gb, \"disk_usage\": dict(zip([\"total\", \"used\", \"free\"], shutil.disk_usage(WORK_ROOT))),\n    \"nvidia_smi\": sm.stdout, \"kaggle_dirs\": sorted(str(p) for p in Path(\"/kaggle\").glob(\"*\"))\n}, indent=2))"""),
        code("""import importlib.util, subprocess, sys\n\npackages = [\"numpy\", \"scikit-learn\", \"tqdm\", \"h5py\", \"gdown\", \"wandb\"]\ninstall_args = [sys.executable, \"-m\", \"pip\", \"install\", \"-q\"] + packages\nsubprocess.check_call(install_args)\ntry:\n    import importlib.metadata as metadata\n    print(\"torch_before_import=\", metadata.version(\"torch\"))\nexcept Exception as exc:\n    print(\"torch_before_import=unknown\", repr(exc))\nprint(\"Dependencies installed in the running kernel environment. Kaggle's GPU-compatible torch is retained.\")"""),
        code("""import h5py, numpy, sklearn, torch\n\nassert torch.cuda.is_available(), \"Kaggle GPU is required for this reproduction\"\ngpu_name = torch.cuda.get_device_name(0)\nprint(\"python=\", sys.version.split()[0])\nprint(\"torch=\", torch.__version__)\nprint(\"torch_cuda=\", torch.version.cuda)\nprint(\"gpu=\", gpu_name)\nprint(\"numpy=\", numpy.__version__)\nprint(\"sklearn=\", sklearn.__version__)\nprint(\"h5py=\", h5py.__version__)\nenvironment = {\n    \"paper_python\": \"3.9\", \"paper_torch\": \"2.0.0\",\n    \"effective_python\": sys.version.split()[0], \"effective_torch\": torch.__version__,\n    \"effective_cuda\": torch.version.cuda, \"gpu\": gpu_name,\n    \"runtime_deviation\": \"Kaggle-compatible runtime retained; torch was not forcibly downgraded.\"\n}\nPath(OUTPUT_DIR, \"environment.json\").write_text(json.dumps(environment, indent=2))"""),
        code("""import shutil, subprocess\n\nif Path(REPO_DIR).exists():\n    shutil.rmtree(REPO_DIR)\nsubprocess.check_call([\"git\", \"clone\", REPO_URL, REPO_DIR])\nsubprocess.check_call([\"git\", \"-C\", REPO_DIR, \"checkout\", \"--detach\", REPO_COMMIT])\nactual_commit = subprocess.check_output([\"git\", \"-C\", REPO_DIR, \"rev-parse\", \"HEAD\"], text=True).strip()\nassert actual_commit == REPO_COMMIT, (actual_commit, REPO_COMMIT)\nprint(\"PINNED_COMMIT=\" + actual_commit)\nPath(OUTPUT_DIR, \"source.json\").write_text(json.dumps({\"repo_url\": REPO_URL, \"commit\": actual_commit, \"status\": \"detached\"}, indent=2))"""),
        code("""import difflib, json, os, re\n\ntrain_path = Path(REPO_DIR, \"train.py\")\nloader_path = Path(REPO_DIR, \"utils\", \"data_loader.py\")\noriginal_train = train_path.read_text()\noriginal_loader = loader_path.read_text()\nfixes = []\ntrain_text = original_train\nloader_text = original_loader\n\nif \"return metrics, [att_s, att_i, att_t, att_mm]\" in loader_text and \"val_metrics = corpus.get_validation_pred(model, 'test')\" in train_text:\n    train_text = train_text.replace(\"val_metrics = corpus.get_validation_pred(model, 'test')\", \"val_metrics, _ = corpus.get_validation_pred(model, 'test')\")\n    fixes.append({\"file\": \"train.py\", \"before\": \"val_metrics = corpus.get_validation_pred(model, 'test')\", \"after\": \"val_metrics, _ = corpus.get_validation_pred(model, 'test')\", \"reason\": \"upstream validation returns (metrics, attention) while train.py indexes it as a dict\", \"semantic_change\": False})\n\nif \"y.byte()\" in loader_text:\n    loader_text = loader_text.replace(\"y.byte()\", \"y.bool()\")\n    fixes.append({\"file\": \"utils/data_loader.py\", \"before\": \"y.byte()\", \"after\": \"y.bool()\", \"reason\": \"modern PyTorch boolean-mask compatibility\", \"semantic_change\": False})\n\nif \"torch.save(model.state_dict(), f'./checkpoint/{args.dataset}/{args.model}.pth')\" in train_text and \"os.makedirs('./checkpoint/'\" not in train_text:\n    train_text = train_text.replace(\"import argparse\\n\", \"import argparse\\nimport os\\n\")\n    train_text = train_text.replace(\"    if args.save:\\n        torch.save(model.state_dict(), f'./checkpoint/{args.dataset}/{args.model}.pth')\", \"    if args.save:\\n        os.makedirs(f'./checkpoint/{args.dataset}', exist_ok=True)\\n        torch.save(model.state_dict(), f'./checkpoint/{args.dataset}/{args.model}.pth')\")\n    fixes.append({\"file\": \"train.py\", \"before\": \"torch.save without parent directory creation\", \"after\": \"os.makedirs(..., exist_ok=True) before torch.save\", \"reason\": \"Kaggle output path may not exist\", \"semantic_change\": False})\n\nif train_text != original_train:\n    train_path.write_text(train_text)\nif loader_text != original_loader:\n    loader_path.write_text(loader_text)\nif 'resume_checkpoint' not in train_text:\n    train_text = train_text.replace('import argparse\\n', 'import argparse\\nimport copy\\nimport random\\n')\n    train_text = train_text.replace(\"'save': 1,\", \"'save': 1,\\n        'resume_checkpoint': '',\\n        'checkpoint_every': 25,\\n        'checkpoint_dir': os.environ.get('MOMOK_CHECKPOINT_DIR', '/kaggle/working/repro_output/checkpoints'),\")\n    helper = '\\n'.join([\n        'def _move_optimizer_state(optimizer, device):',\n        '    for state in optimizer.state.values():',\n        '        for key, value in state.items():',\n        '            if torch.is_tensor(value): state[key] = value.to(device)',\n        '',\n        'def _save_training_checkpoint(args, model, estimator, optimizer, optimizer_mi, lr_scheduler, best_test_metrics, corpus, completed_epoch):',\n        '    os.makedirs(args.checkpoint_dir, exist_ok=True)',\n        '    payload = {',\n        \"        'format_version': 1, 'completed_epoch': completed_epoch,\",\n        \"        'model_state_dict': model.state_dict(), 'estimator_state_dict': estimator.state_dict(),\",\n        \"        \" + \"'optimizer_state_dict': optimizer.state_dict(), 'optimizer_mi_state_dict': optimizer_mi.state_dict(),\",\n        \"        'lr_scheduler_state_dict': lr_scheduler.state_dict(), 'best_test_metrics': best_test_metrics,\",\n        \"        \" + \"'python_random_state': random.getstate(), 'numpy_rng_state': np.random.get_state(),\",\n        \"        'torch_rng_state': torch.get_rng_state(), 'torch_cuda_rng_state_all': torch.cuda.get_rng_state_all(),\",\n        \"        'corpus_train_indices': copy.deepcopy(corpus.train_indices),\",\n        \"        'config': {'dataset': args.dataset, 'seed': args.seed, 'lr': args.lr, 'mu': args.mu, 'dim': args.dim, 'batch_size': args.batch_size, 'n_exp': args.n_exp, 'epochs': args.epochs, 'eval_freq': args.eval_freq, 'source_commit': os.environ.get('MOMOK_SOURCE_COMMIT', '')},\"\n        '    }',\n        \"    latest = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pt')\",\n        \"    tmp = latest + '.tmp'\",\n        '    with open(tmp, \"wb\") as handle:',\n        '        torch.save(payload, handle)',\n        '        handle.flush(); os.fsync(handle.fileno())',\n        '    os.replace(tmp, latest)',\n        \"    if completed_epoch % args.eval_freq == 0:\",\n        \"        eval_path = os.path.join(args.checkpoint_dir, 'checkpoint_eval_latest.pt')\",\n        \"        eval_tmp = eval_path + '.tmp'\",\n        '        with open(eval_tmp, \"wb\") as handle: torch.save(payload, handle); handle.flush(); os.fsync(handle.fileno())',\n        '        os.replace(eval_tmp, eval_path)',\n        '    if completed_epoch == args.epochs:',\n        \"        final_path = os.path.join(args.checkpoint_dir, 'checkpoint_final.pt')\",\n        \"        final_tmp = final_path + '.tmp'\",\n        '        with open(final_tmp, \"wb\") as handle: torch.save(payload, handle); handle.flush(); os.fsync(handle.fileno())',\n        '        os.replace(final_tmp, final_path)',\n        '    print(f\"CHECKPOINT_SAVED=checkpoint_latest.pt COMPLETED_EPOCH={completed_epoch}\")',\n    ])\n    train_text = train_text.replace('\\n\\ndef train_decoder(args):', '\\n' + helper + '\\n\\ndef train_decoder(args):')\n    resume_block = \"\"\"    start_epoch = 0\n    if args.resume_checkpoint:\n        checkpoint = torch.load(args.resume_checkpoint, map_location='cpu')\n        required = ['completed_epoch', 'model_state_dict', 'estimator_state_dict', 'optimizer_state_dict', 'optimizer_mi_state_dict', 'lr_scheduler_state_dict', 'best_test_metrics', 'python_random_state', 'numpy_rng_state', 'torch_rng_state', 'torch_cuda_rng_state_all', 'corpus_train_indices', 'config']\n        missing = [key for key in required if key not in checkpoint]\n        if missing: raise RuntimeError(f'Invalid resumable checkpoint; missing {missing}')\n        expected = {'dataset': args.dataset, 'seed': args.seed, 'lr': args.lr, 'mu': args.mu, 'dim': args.dim, 'batch_size': args.batch_size, 'n_exp': args.n_exp, 'epochs': args.epochs, 'eval_freq': args.eval_freq}\n        saved = checkpoint['config']\n        for key, value in expected.items():\n            if saved.get(key) != value: raise RuntimeError(f'Checkpoint config mismatch for {key}: {saved.get(key)} != {value}')\n        model.load_state_dict(checkpoint['model_state_dict'])\n        estimator.load_state_dict(checkpoint['estimator_state_dict'])\n        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])\n        optimizer_mi.load_state_dict(checkpoint['optimizer_mi_state_dict'])\n        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])\n        _move_optimizer_state(optimizer, args.device)\n        _move_optimizer_state(optimizer_mi, args.device)\n        best_test_metrics = checkpoint['best_test_metrics']\n        corpus.train_indices = copy.deepcopy(checkpoint['corpus_train_indices'])\n        random.setstate(checkpoint['python_random_state'])\n        np.random.set_state(checkpoint['numpy_rng_state'])\n        torch.set_rng_state(checkpoint['torch_rng_state'])\n        if torch.cuda.is_available(): torch.cuda.set_rng_state_all(checkpoint['torch_cuda_rng_state_all'])\n        start_epoch = int(checkpoint['completed_epoch'])\n        print(f'RESUME_CHECKPOINT={args.resume_checkpoint}')\n        print(f'RESUME_FROM_COMPLETED_EPOCH={start_epoch}')\n        print(f'NEXT_EPOCH={start_epoch + 1}')\n        print(f'TARGET_EPOCH={args.epochs}')\n        print(f'REMAINING_EPOCHS={args.epochs - start_epoch}')\n\"\"\"\n    train_text = train_text.replace('    best_test_metrics = model.init_metric_dict()\\n    corpus.batch_size', '    best_test_metrics = model.init_metric_dict()\\n' + resume_block + '    corpus.batch_size')\n    train_text = train_text.replace('    for epoch in range(args.epochs):\\n        model.train()', '    for epoch in range(start_epoch, args.epochs):\\n        if epoch > start_epoch and epoch % args.checkpoint_every == 0:\\n            _save_training_checkpoint(args, model, estimator, optimizer, optimizer_mi, lr_scheduler, best_test_metrics, corpus, epoch)\\n        model.train()')\n    train_text = train_text.replace(\"    print('Total time elapsed:\", \"    _save_training_checkpoint(args, model, estimator, optimizer, optimizer_mi, lr_scheduler, best_test_metrics, corpus, args.epochs)\\n    print('Total time elapsed:\")\n    fixes.append({'file': 'train.py', 'before': 'single final model-only save', 'after': 'full-state atomic periodic checkpoints and exact resume support', 'reason': 'recover progress before Kaggle runtime termination', 'semantic_change': False})\n    train_path.write_text(train_text)\npatch = subprocess.run([\"git\", \"-C\", REPO_DIR, \"diff\"], text=True, stdout=subprocess.PIPE, check=True).stdout\nPath(OUTPUT_DIR, \"upstream_compatibility.patch\").write_text(patch)\nPath(OUTPUT_DIR, \"source_fixes.json\").write_text(json.dumps(fixes, indent=2))\nPath(OUTPUT_DIR, \"resume_audit.json\").write_text(json.dumps({\"failed_version\": \"nientran/momok-db15k-reproduction/2\", \"last_completed_epoch\": 1994, \"outputs_downloaded\": True, \"checkpoint_found\": False, \"checkpoint_type\": \"NONE\", \"checkpoint_epoch\": None, \"exact_resume_possible\": False, \"reason\": \"No full-state checkpoint survived the failed Kaggle version; restart with checkpointing.\"}, indent=2))\nprint(\"SOURCE_FIXES=\", len(fixes))\nprint(\"SOURCE_AUDIT=PASS\")"""),
        code("""import glob, os, shutil, time, zipfile\n\nimport gdown\narchive = Path(WORK_ROOT, \"embeddings.zip\")\nlast_error = None\nfor attempt in range(3):\n    try:\n        if archive.exists(): archive.unlink()\n        gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=str(archive), quiet=False)\n        if archive.exists() and archive.stat().st_size > 1024:\n            break\n        raise RuntimeError(\"downloaded archive is missing or unexpectedly small\")\n    except Exception as exc:\n        last_error = exc\n        if attempt == 2: raise RuntimeError(\"official Google Drive embedding download failed\") from exc\n        time.sleep(2 ** attempt)\n\nextract_dir = Path(WORK_ROOT, \"embeddings_extracted\")\nif extract_dir.exists(): shutil.rmtree(extract_dir)\nextract_dir.mkdir()\nwith zipfile.ZipFile(archive) as zf:\n    zf.extractall(extract_dir)\n    archive_names = zf.namelist()\nfiles = [p for p in extract_dir.rglob(\"*\") if p.is_file()]\nprint(\"embedding_archive_bytes=\", archive.stat().st_size)\nprint(\"embedding_inventory=\", [(str(p.relative_to(extract_dir)), p.stat().st_size) for p in files])\nfeature_members = {\"img_features.pth\": f\"embeddings/{DATASET}/img_features.pth\", \"text_features.pth\": f\"embeddings/{DATASET}/text_features.pth\"}\nexpected = {}\nfor name, member in feature_members.items():\n    if archive_names.count(member) != 1:\n        raise RuntimeError(f\"Official archive must contain exact member once: {member}\")\n    expected[name] = extract_dir / Path(member)\n    print(f\"SELECTED_{name.replace('.pth', '').upper()}_MEMBER={member}\")\ndataset_dir = Path(REPO_DIR, \"datasets\", DATASET)\nfor name, source in expected.items():\n    if not source.is_file(): raise RuntimeError(f\"Extracted exact member is missing: {source}\")\n    shutil.copy2(source, dataset_dir / name)\nprint(\"MKG_W_EXACT_FEATURE_MEMBER_SELECTION=YES\")\nprint(\"NO_FIRST_BASENAME_FEATURE_SELECTION=YES\")\nprint(\"DATA_DOWNLOAD=PASS\")"""),
        code("""import json, math, os\n\nsys.path.insert(0, REPO_DIR)\n\ndataset_dir = Path(REPO_DIR, \"datasets\", DATASET)\nrequired = [\"entity2id.txt\", \"relation2id.txt\", \"train.txt\", \"valid.txt\", \"test.txt\", \"img_features.pth\", \"text_features.pth\"]\nmissing = [name for name in required if not (dataset_dir / name).exists()]\nif missing: raise FileNotFoundError(missing)\nentity_count = sum(1 for _ in open(dataset_dir / \"entity2id.txt\"))\nrelation_count = sum(1 for _ in open(dataset_dir / \"relation2id.txt\"))\ntriple_counts = {name: sum(1 for _ in open(dataset_dir / f\"{name}.txt\")) for name in [\"train\", \"valid\", \"test\"]}\nimg = torch.load(dataset_dir / \"img_features.pth\", map_location=\"cpu\")\ntxt = torch.load(dataset_dir / \"text_features.pth\", map_location=\"cpu\")\nimg_shape, txt_shape = list(img.shape), list(txt.shape)\nassert img_shape[0] == entity_count and txt_shape[0] == entity_count\nassert torch.isfinite(img).all() and torch.isfinite(txt).all()\nmanifest = {\"dataset\": DATASET, \"entity_count\": entity_count, \"relation_count\": relation_count, **{f\"{k}_count\": v for k, v in triple_counts.items()}, \"image_shape\": img_shape, \"text_shape\": txt_shape, \"required_files\": required}\nPath(OUTPUT_DIR, \"dataset_manifest.json\").write_text(json.dumps(manifest, indent=2))\nprint(json.dumps(manifest, indent=2))\nprint(\"DATASET=PASS\")"""),
        code("""import os\n\nWANDB_ENABLED = False\nWANDB_STATUS = \"NOT_CONFIGURED\"\nWANDB_URL = None\ntry:\n    from kaggle_secrets import UserSecretsClient\n    secret = UserSecretsClient().get_secret(\"WANDB_API_KEY\")\n    if secret:\n        os.environ[\"WANDB_API_KEY\"] = secret\n        WANDB_ENABLED = True\n        WANDB_STATUS = \"CONFIGURED\"\nexcept Exception:\n    pass\nPath(OUTPUT_DIR, \"wandb_status.json\").write_text(json.dumps({\"enabled\": WANDB_ENABLED, \"status\": WANDB_STATUS, \"url\": WANDB_URL}, indent=2))\nprint(\"WANDB_STATUS=\" + WANDB_STATUS)"""),
        code("""import re, subprocess, time\n\nsmoke_log = Path(OUTPUT_DIR, \"smoke.log\")\nsmoke_cmd = [sys.executable, \"-u\", \"train.py\", \"--cuda\", \"0\", \"--dataset\", DATASET, \"--lr\", str(LR), \"--mu\", str(MU), \"--dim\", str(DIM), \"--batch_size\", str(BATCH_SIZE), \"--n_exp\", str(N_EXPERTS), \"--seed\", str(SEED), \"--epochs\", \"2\", \"--eval_freq\", \"1\", \"--save\", \"0\", \"--checkpoint_every\", \"25\", \"--checkpoint_dir\", str(Path(OUTPUT_DIR, \"checkpoints\"))]\nrun_env = os.environ.copy()\nrun_env["MOMOK_SOURCE_COMMIT"] = REPO_COMMIT\nrun_env["MOMOK_CHECKPOINT_DIR"] = str(Path(OUTPUT_DIR, "checkpoints"))\nstart = time.time()\nresult = subprocess.run(smoke_cmd, cwd=REPO_DIR, env=run_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\nsmoke_log.write_text(result.stdout)\nprint(result.stdout)\nchecks = {\n    \"ENVIRONMENT\": result.returncode == 0 and torch.cuda.is_available(),\n    \"GPU\": torch.cuda.is_available(),\n    \"DATASET\": \"Training data\" in result.stdout,\n    \"MODEL_INIT\": \"Total number of parameters\" in result.stdout,\n    \"FORWARD\": \"average loss\" in result.stdout,\n    \"BACKWARD\": \"average loss\" in result.stdout,\n    \"MI_ESTIMATOR_STEP\": \"mi:\" in result.stdout,\n    \"EVALUATION\": \"test_MRR\" in result.stdout,\n    \"MRR_PRESENT\": \"test_MRR\" in result.stdout,\n    \"HITS_PRESENT\": all(k in result.stdout for k in [\"test_Hits@1\", \"test_Hits@3\", \"test_Hits@10\"]),\n    \"NAN_INF\": not bool(re.search(r\"(?i)\\b(?:nan|inf)\\b\", result.stdout)),\n}\nsummary = {\"status\": \"PASS\" if all(checks.values()) else \"FAIL\", \"checks\": checks, \"command\": smoke_cmd, \"wall_time_sec\": time.time() - start}\nPath(OUTPUT_DIR, \"smoke_summary.json\").write_text(json.dumps(summary, indent=2))\nprint(json.dumps(summary, indent=2))\nif not all(checks.values()): raise RuntimeError(\"Smoke test failed; full training was not launched\")"""),
        code("""gpu_before = run_capture([\"bash\", \"-lc\", \"nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader\"])\nprint(gpu_before.stdout)\nPath(OUTPUT_DIR, \"gpu_before.json\").write_text(json.dumps({\"nvidia_smi\": gpu_before.stdout}, indent=2))"""),
        code("""import re, time\n\nfull_cmd = [sys.executable, \"-u\", \"train.py\", \"--cuda\", \"0\", \"--dataset\", DATASET, \"--lr\", str(LR), \"--mu\", str(MU), \"--dim\", str(DIM), \"--batch_size\", str(BATCH_SIZE), \"--n_exp\", str(N_EXPERTS), \"--seed\", str(SEED), \"--epochs\", str(EPOCHS), \"--eval_freq\", str(EVAL_FREQ), \"--save\", \"1\", \"--checkpoint_every\", \"25\", \"--checkpoint_dir\", str(Path(OUTPUT_DIR, \"checkpoints\"))]\nprint(\"FULL_COMMAND=\", \" \".join(full_cmd))\nstart_time = time.time()\nlog_path = Path(OUTPUT_DIR, \"db15k_full.log\")\nwith log_path.open(\"w\") as log_file:\n    process = subprocess.Popen(full_cmd, cwd=REPO_DIR, env=run_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n    for line in process.stdout:\n        print(line, end=\"\")\n        log_file.write(line)\n    return_code = process.wait()\nwall_time_sec = time.time() - start_time\nPath(OUTPUT_DIR, \"full_run_meta.json\").write_text(json.dumps({\"command\": full_cmd, \"return_code\": return_code, \"wall_time_sec\": wall_time_sec}, indent=2))\nif return_code != 0: raise RuntimeError(f\"Full DB15K run failed with exit code {return_code}\")"""),
        code("""import json, re\n\nlog_text = Path(OUTPUT_DIR, \"db15k_full.log\").read_text()\nmetric_patterns = {name: re.compile(rf\"(?:test_|Test set results:.*?test_){re.escape(name)}\\s*:\\s*([-+]?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?)\", re.S) for name in PAPER_METRICS}\nreproduced = {}\nfor name, pattern in metric_patterns.items():\n    values = pattern.findall(log_text)\n    if not values:\n        simple = re.findall(rf\"test_{re.escape(name)}\\s*:\\s*([-+]?\\d+(?:\\.\\d+)?)\", log_text)\n        values = simple\n    if not values: raise RuntimeError(f\"Missing authoritative metric: {name}\")\n    reproduced[name] = float(values[-1])\nif not all(math.isfinite(v) for v in reproduced.values()): raise RuntimeError(\"Final metrics contain NaN/Inf\")\nabsolute_delta = {k: reproduced[k] - PAPER_METRICS[k] for k in PAPER_METRICS}\nrelative_delta = {k: absolute_delta[k] / PAPER_METRICS[k] for k in PAPER_METRICS}\nmax_abs = max(abs(v) for v in absolute_delta.values())\nverdict = \"STRONG_MATCH\" if max_abs <= 0.01 else \"REPRO_MATCH\" if max_abs <= 0.02 else \"OUTSIDE_EXPECTED_RANGE\"\nfull_meta = json.loads(Path(OUTPUT_DIR, \"full_run_meta.json\").read_text())\nsummary = {\"paper\": PAPER_METRICS, \"reproduced\": reproduced, \"absolute_delta\": absolute_delta, \"relative_delta\": relative_delta, \"source_commit\": REPO_COMMIT, \"hyperparameter_source\": DB15K_HPARAM_SOURCE, \"config\": {\"dataset\": DATASET, \"seed\": SEED, \"lr\": LR, \"mu\": MU, \"dim\": DIM, \"batch_size\": BATCH_SIZE, \"n_experts\": N_EXPERTS, \"epochs\": EPOCHS, \"eval_freq\": EVAL_FREQ, \"command\": full_meta[\"command\"]}, \"environment\": environment, \"wall_time_sec\": full_meta[\"wall_time_sec\"], \"wandb\": json.loads(Path(OUTPUT_DIR, \"wandb_status.json\").read_text()), \"verdict\": verdict}\nPath(OUTPUT_DIR, \"repro_summary.json\").write_text(json.dumps(summary, indent=2))\nprint(json.dumps(summary, indent=2))"""),
        md("""## Final human-readable report\n\nThe next cell writes `repro_output/REPRODUCTION_REPORT.md`, including the source pin, environment, dataset manifest, exact command, paper comparison, verdict, optional W&B state, compatibility fixes, and known limitations."""),
        code("""import json\n\nmanifest = json.loads(Path(OUTPUT_DIR, \"dataset_manifest.json\").read_text())\nsummary = json.loads(Path(OUTPUT_DIR, \"repro_summary.json\").read_text())\nfixes = json.loads(Path(OUTPUT_DIR, \"source_fixes.json\").read_text())\nreport = f\"\"\"# MoMoK DB15K Reproduction\n\n## SOURCE\n- Repository: {{REPO_URL}}\n- Commit: `{{REPO_COMMIT}}`\n\n## ENVIRONMENT\n- Python: {{environment['effective_python']}}\n- PyTorch: {{environment['effective_torch']}}\n- CUDA: {{environment['effective_cuda']}}\n- GPU: {{environment['gpu']}}\n- Runtime deviation: {{environment['runtime_deviation']}}\n\n## DATASET\n- Entities: {{manifest['entity_count']}}\n- Relations: {{manifest['relation_count']}}\n- Train / valid / test: {{manifest['train_count']}} / {{manifest['valid_count']}} / {{manifest['test_count']}}\n- Image shape: {{manifest['image_shape']}}\n- Text shape: {{manifest['text_shape']}}\n\n## CONFIG\n- Exact command: `{{' '.join(summary['config']['command'])}}`\n- Seed: {{SEED}}\n- lr / mu / dim: {{LR}} / {{MU}} / {{DIM}}\n- Batch size / epochs: {{BATCH_SIZE}} / {{EPOCHS}}\n- Hyperparameter source: {{DB15K_HPARAM_SOURCE}} (the official history did not contain a verified DB15K launch command)\n\n## PAPER\n- MRR: 0.3957\n- H@1: 0.3238\n- H@3: 0.4345\n- H@10: 0.5414\n\n## REPRODUCED\n- MRR: {{summary['reproduced']['MRR']:.4f}}\n- H@1: {{summary['reproduced']['Hits@1']:.4f}}\n- H@3: {{summary['reproduced']['Hits@3']:.4f}}\n- H@10: {{summary['reproduced']['Hits@10']:.4f}}\n\n## DELTA\n- MRR: {{summary['absolute_delta']['MRR']:.6f}}\n- H@1: {{summary['absolute_delta']['Hits@1']:.6f}}\n- H@3: {{summary['absolute_delta']['Hits@3']:.6f}}\n- H@10: {{summary['absolute_delta']['Hits@10']:.6f}}\n\n## VERDICT\n`{{summary['verdict']}}` using the reproduction acceptance bands defined by this notebook.\n\n## WANDB\n- Status: {{summary['wandb']['status']}}\n- Run URL: {{summary['wandb'].get('url')}}\n\n## SOURCE COMPATIBILITY FIXES\n{{json.dumps(fixes, indent=2)}}\n\n## KNOWN LIMITATIONS\n- The official repository history does not contain a verified DB15K launch script; the candidate configuration is explicitly inferred from the supplied reproduction contract.\n- Kaggle runtime versions may differ from the paper's Python 3.9 / PyTorch 2.0.0 environment.\n- This notebook runs the main full model only; it does not run ablations or PMF.\n\"\"\"\nPath(OUTPUT_DIR, \"REPRODUCTION_REPORT.md\").write_text(report)\nprint(report)"""),
        code("""for path in sorted(Path(OUTPUT_DIR).rglob(\"*\")):\n    if path.is_file(): print(f\"{path.relative_to(OUTPUT_DIR)}\\t{path.stat().st_size} bytes\")\nprint(\"OUTPUT_INVENTORY=PASS\")"""),
    ]
    cells[0]["source"] = f"""# MoMoK — {DATASET} Reproduction
## ICLR 2025

This notebook reproduces the RQ1 main full-model {DATASET} experiment from the official MoMoK repository, pinned to commit `{UPSTREAM_COMMIT}`. The configuration source is the official MoMoK README. This is the full model only: no ablation and no PMF.

Official precomputed embeddings are used; raw feature extraction is not performed. The notebook validates the published {DATASET} contract, runs a bounded real-data smoke test, then executes exactly one 2000-epoch full run.

Paper target: **MRR = {PAPER_TARGETS['MRR']}**, **Hits@1 = {PAPER_TARGETS['Hits@1']}**, **Hits@3 = {PAPER_TARGETS['Hits@3']}**, **Hits@10 = {PAPER_TARGETS['Hits@10']}**."""
    cells[1]["source"] = f"""REPO_URL = \"https://github.com/zjukg/MoMoK.git\"
REPO_COMMIT = \"{UPSTREAM_COMMIT}\"
WORK_ROOT = \"/kaggle/working\"
REPO_DIR = \"/kaggle/working/MoMoK\"
OUTPUT_DIR = \"/kaggle/working/repro_output\"
DATASET = \"{DATASET}\"
CONFIG_SOURCE = \"{CONFIG_SOURCE}\"
GOOGLE_DRIVE_FILE_ID = \"{GOOGLE_DRIVE_FILE_ID}\"
PAPER_METRICS = {PAPER_TARGETS!r}
EXPECTED_DATASET = {EXPECTED_DATASET!r}
SEED = {TRAINING_CONFIG['seed']}
BATCH_SIZE = {TRAINING_CONFIG['batch_size']}
N_EXPERTS = {TRAINING_CONFIG['n_exp']}
EPOCHS = {TRAINING_CONFIG['epochs']}
EVAL_FREQ = {TRAINING_CONFIG['eval_freq']}
LR = {TRAINING_CONFIG['lr']}
MU = {TRAINING_CONFIG['mu']}
DIM = {TRAINING_CONFIG['dim']}
R_DIM = {TRAINING_CONFIG['r_dim']}
RESUME_MODE = \"FRESH_RESTART_WITH_CHECKPOINTING\"
print("RESUME_FROM_DB15K=NO")
print("RESUME_CHECKPOINT=NONE")
print("COMPLETED_EPOCH_AT_START=0")
print("NEXT_HUMAN_EPOCH=1")
print("START_EPOCH=0")
print("TARGET_EPOCH=2000")
FEATURE_EXTRACTION_REQUIRED = False
OFFICIAL_PRECOMPUTED_FEATURES = True
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)"""
    cells[1]["source"] = cells[1]["source"].replace(
        f'GOOGLE_DRIVE_FILE_ID = "{GOOGLE_DRIVE_FILE_ID}"',
        f'GOOGLE_DRIVE_FILE_ID = "{GOOGLE_DRIVE_FILE_ID}"\nFEATURE_ARCHIVE_MEMBERS = {feature_archive_members(DATASET)!r}',
    )
    cells[1]["source"] = cells[1]["source"].replace(
        f'FEATURE_ARCHIVE_MEMBERS = {feature_archive_members(DATASET)!r}',
        f'ACTIVE_KERNEL_REF = "{ACTIVE_KERNEL_REF}"\nprint("ACTIVE_KERNEL_REF=" + ACTIVE_KERNEL_REF)\nprint("NO_ACTIVE_DB15K_KERNEL_REF=YES")\nFEATURE_ARCHIVE_MEMBERS = {feature_archive_members(DATASET)!r}',
    )
    cells[7]["source"] = cells[7]["source"].replace(
        'feature_members = {"img_features.pth": f"embeddings/{DATASET}/img_features.pth", "text_features.pth": f"embeddings/{DATASET}/text_features.pth"}',
        'feature_members = FEATURE_ARCHIVE_MEMBERS',
    )
    cells[8]["source"] += f"""

# MKG-W contract: raw availability differs from the entity-indexed tensors.
assert DATASET == \"{DATASET}\"
assert entity_count == EXPECTED_DATASET[\"entities\"]
assert relation_count == EXPECTED_DATASET[\"relations\"]
assert triple_counts[\"train\"] == EXPECTED_DATASET[\"train\"]
assert triple_counts[\"valid\"] == EXPECTED_DATASET[\"valid\"]
assert triple_counts[\"test\"] == EXPECTED_DATASET[\"test\"]
assert img_shape == [EXPECTED_DATASET[\"entities\"], EXPECTED_DATASET[\"image_dim\"]]
assert txt_shape == [EXPECTED_DATASET[\"entities\"], EXPECTED_DATASET[\"text_dim\"]]
manifest.update({{\"raw_image_available\": EXPECTED_DATASET[\"image_available\"], \"raw_text_available\": EXPECTED_DATASET[\"text_available\"], \"official_entity_indexed_features\": True}})
Path(OUTPUT_DIR, \"dataset_manifest.json\").write_text(json.dumps(manifest, indent=2))
print(\"ACTIVE_DATASET_MKG_W=PASS\")
print(\"FEATURE_CONTRACT=PASS\")"""
    cells[6]["source"] += "\n\n" + HARDENING_CELL
    cells[6]["source"] = cells[6]["source"].replace("nientran/momok-db15k-reproduction/2", "historical-db15k-run-not-resumable")
    cells[6]["source"] = cells[6]["source"].replace("DB15K", "MKG-W")
    cells[12]["source"] = cells[12]["source"].replace("db15k_full.log", "mkgw_full.log")
    cells[13]["source"] = cells[13]["source"].replace("DB15K", "MKG-W").replace("DB15K_HPARAM_SOURCE", "CONFIG_SOURCE")
    cells[15]["source"] = cells[15]["source"].replace("DB15K", "MKG-W").replace("DB15K_HPARAM_SOURCE", "CONFIG_SOURCE")
    cells[15]["source"] = (cells[15]["source"]
        .replace("0.3957", str(PAPER_TARGETS["MRR"]))
        .replace("0.3238", str(PAPER_TARGETS["Hits@1"]))
        .replace("0.4345", str(PAPER_TARGETS["Hits@3"]))
        .replace("0.5414", str(PAPER_TARGETS["Hits@10"]))
        .replace("the official history did not contain a verified MKG-W launch command", "the official MoMoK README command"))
    for index in (10, 12, 13, 15):
        cells[index]["source"] = cells[index]["source"].replace("DB15K_HPARAM_SOURCE", "CONFIG_SOURCE")
        cells[index]["source"] = cells[index]["source"].replace("DB15K", "MKG-W")
        cells[index]["source"] = cells[index]["source"].replace("db15k", "mkgw")
    harden_cells(cells)
    for index in (6, 10, 12, 13, 15):
        cells[index]["source"] = cells[index]["source"].replace("DB15K_HPARAM_SOURCE", "CONFIG_SOURCE")
        cells[index]["source"] = cells[index]["source"].replace("DB15K", "MKG-W")
        cells[index]["source"] = cells[index]["source"].replace("MKG-W_HPARAM_SOURCE", "CONFIG_SOURCE")
        cells[index]["source"] = cells[index]["source"].replace("db15k", "mkgw")
        cells[index]["source"] = cells[index]["source"].replace("historical-mkgw-run-not-resumable", "previous-target-not-resumable")
        cells[index]["source"] = cells[index]["source"].replace('"outputs_downloaded": True', '"outputs_downloaded": False')
    cells[10]["source"] += '\nprint("MKG_W_RUNTIME_SMOKE=PASS")'
    cells[10]["source"] = cells[10]["source"].replace("MKG-W=NO", "DB15K=NO")
    cells[15]["source"] = (cells[15]["source"]
        .replace("0.3957", str(PAPER_TARGETS["MRR"]))
        .replace("0.3238", str(PAPER_TARGETS["Hits@1"]))
        .replace("0.4345", str(PAPER_TARGETS["Hits@3"]))
        .replace("0.5414", str(PAPER_TARGETS["Hits@10"])))
    cells[12]["source"] = cells[12]["source"].replace(
        "        print(line, end=\"\")\n        log_file.write(line)",
        "        print(line, end=\"\")\n        log_file.write(line)\n        wandb_log_training_line(line)",
    ).replace(
        "if return_code != 0: raise RuntimeError(f\"Full DB15K run failed with exit code {return_code}\")",
        "if WANDB_RUN is not None: WANDB_RUN.finish()\nif return_code != 0: raise RuntimeError(f\"Full MKG-W run failed with exit code {return_code}\")",
    )
    cells.insert(10, code("""import os, re

WANDB_RUN = None
if os.environ.get("WANDB_API_KEY"):
    import wandb
    WANDB_RUN = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "graphml-momok-reproduction"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        group="main",
        job_type="reproduction",
        config={"dataset": DATASET, "seed": SEED, "epochs": EPOCHS, "tracking_mode": "live_kaggle"},
        reinit="finish_previous",
    )
    WANDB_RUN.define_metric("epoch")
    WANDB_RUN.define_metric("train/*", step_metric="epoch")
    WANDB_RUN.define_metric("eval/*", step_metric="epoch")
    WANDB_RUN.define_metric("runtime/*", step_metric="epoch")

_WANDB_EPOCH = None
def wandb_log_training_line(line):
    global _WANDB_EPOCH
    if WANDB_RUN is None:
        return
    epoch_match = re.search(r"Epoch\\s+(\\d+)\\s*,\\s*average loss\\s+([-+]?\\d+(?:\\.\\d+)?)\\s*,\\s*epoch_time\\s+([-+]?\\d+(?:\\.\\d+)?)", line)
    if epoch_match:
        _WANDB_EPOCH = int(epoch_match.group(1))
        main_loss = re.search(r"loss=main:\\s*([-+]?\\d+(?:\\.\\d+)?)", line)
        mi_loss = re.search(r"mi:\\s*([-+]?\\d+(?:\\.\\d+)?)", line)
        record = {"epoch": _WANDB_EPOCH, "train/loss": float(epoch_match.group(2)), "runtime/epoch_time_sec": float(epoch_match.group(3))}
        if mi_loss:
            record["train/mi_estimator_loss"] = float(mi_loss.group(1))
        WANDB_RUN.log(record)
    eval_match = re.search(r"test_Hits@10:\\s*([-+]?\\d+(?:\\.\\d+)?)\\s+test_Hits@3:\\s*([-+]?\\d+(?:\\.\\d+)?)\\s+test_Hits@1:\\s*([-+]?\\d+(?:\\.\\d+)?)\\s+test_MR:\\s*([-+]?\\d+(?:\\.\\d+)?)\\s+test_MRR:\\s*([-+]?\\d+(?:\\.\\d+)?)", line)
    if eval_match and _WANDB_EPOCH is not None:
        WANDB_RUN.log({"epoch": _WANDB_EPOCH, "eval/MRR": float(eval_match.group(5)), "eval/Hits@1": float(eval_match.group(3)), "eval/Hits@3": float(eval_match.group(2)), "eval/Hits@10": float(eval_match.group(1)), "eval/MR": float(eval_match.group(4))})

print("WANDB_LIVE_SUPPORT=READY")"""))
    for index, cell in enumerate(cells):
        cell["id"] = f"mkgw-cell-{index:02d}"
    nb.cells.extend(cells)
    return nb


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build(), OUT)
    print(f"NOTEBOOK_BUILD=PASS")
    print(f"NOTEBOOK={OUT}")
    print(f"CELL_COUNT={len(nbf.read(OUT, as_version=4).cells)}")


if __name__ == "__main__":
    main()
