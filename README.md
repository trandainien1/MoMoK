# MoMoK Reproduction

This public branch is a reproduction workflow built on the official
[zjukg/MoMoK](https://github.com/zjukg/MoMoK) repository. The experiments use
the pinned upstream source commit
`99c2df114d48c79708ea0608644b685183d81bd9`.

The original upstream README is preserved as [UPSTREAM_README.md](UPSTREAM_README.md).
This README is intentionally a how-to guide; experiment results and scientific
interpretation belong in the W&B report and course report.

## Reproduction scope

The maintained workflows cover the MoMoK full-model main-result reproductions
for MKG-W and MKG-Y. They do not launch ablations, PMF, or unrelated datasets.

## Prerequisites

- Linux/WSL with Python 3.
- A CUDA-capable GPU for Kaggle execution.
- Git and the Kaggle CLI.
- A W&B account and the W&B SDK for optional publication/finalization.
- Kaggle Internet access for the official source and embedding download.

The notebook keeps Kaggle's GPU-compatible PyTorch runtime and records the
effective Python, PyTorch, CUDA, and GPU versions in its outputs.

## Installation

Clone the public fork and select the stable publication snapshot:

```bash
git clone https://github.com/trandainien1/MoMoK.git
cd MoMoK
git checkout course-momok-main-results-v1
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip nbformat numpy pandas wandb wandb-workspaces
```

The Kaggle CLI is used through the local environment configured by the
operator. Do not commit a virtual environment or credential file.

## Kaggle authentication

Create the ignored local credential surface required by the helper scripts:

```bash
mkdir -p .secrets
chmod 700 .secrets
${EDITOR:-vi} .secrets/kaggle.env
chmod 600 .secrets/kaggle.env
```

The file is local-only and must provide the Kaggle account/token variables
expected by `scripts/kaggle_env.sh`. The helper requires the account
`nientrandai1` for the maintained kernels and exports the repository-local
Kaggle config context. Never put credentials in notebook source, README, Git,
or W&B artifacts.

Before a Kaggle command, source the environment in the current shell:

```bash
set +x
source scripts/kaggle_env.sh
bash scripts/kaggle_auth.sh
```

## W&B setup

The W&B API key is never committed. For a live Kaggle run, add it through the
Kaggle UI:

Kaggle Notebook → Add-ons → Secrets → add `WANDB_API_KEY` → enable notebook
access.

The MKG-Y notebook fails before expensive training if that secret is absent.
The key is retrieved at runtime through Kaggle Secrets. The local W&B helpers
can be used for safe post-run publication with credentials loaded from an
ignored local environment file.

## MKG-W reproduction

From the repository root, build and statically validate the MKG-W notebook:

```bash
python scripts/build_kaggle_notebook.py
python scripts/validate_notebook.py
```

The generated notebook is under `kaggle/momok-db15k/` for historical
compatibility with the maintained builder. It targets MKG-W at runtime and
uses the official precomputed feature members for that dataset. After
reviewing the generated notebook and attaching any required Kaggle secrets,
the maintained Kaggle launcher is:

```bash
bash scripts/kaggle_push_run.sh
```

The launcher rebuilds/validates the notebook, configures private GPU kernel
metadata, and pushes the kernel. Do not start another run solely to compare
against published results.

## MKG-Y reproduction

Build and validate the separate MKG-Y notebook:

```bash
python scripts/build_mkgy_notebook.py
python scripts/validate_mkgy_notebook.py
```

Prepare the Kaggle kernel metadata and push with the maintained MKG-Y wrapper:

```bash
set +x
source scripts/kaggle_env.sh
bash scripts/kaggle_mkgy_push.sh
```

For live W&B tracking, upload/build the notebook, then in Kaggle attach
`WANDB_API_KEY` through Add-ons → Secrets before selecting Save & Run All. The
same logical W&B run ID is reused for a supported checkpoint-resume session;
the secret itself never enters this repository.

## Monitoring and outputs

The MKG-Y monitor uses the fixed repository-local Kaggle auth context:

```bash
bash scripts/kaggle_monitor_mkgy.sh
```

To download a completed kernel's outputs, pass its fully qualified kernel
reference to the generic downloader:

```bash
bash scripts/kaggle_download_outputs.sh nientrandai1/momok-mkg-y-reproduction
```

Downloaded material is written under the ignored `kaggle_outputs/` directory.
The notebook output includes environment and dataset manifests, source/patch
metadata, smoke evidence, training logs, final summaries, and checkpoint
metadata. Large checkpoints and feature tensors are intentionally not part of
the public Git repository.

## Checkpoint and resume

The runtime hardening layer supports atomic full-state checkpoints and exact
continuation from the completed epoch. A compatible checkpoint carries model
and estimator state, both optimizer states, scheduler state, RNG state,
training order state, configuration, dataset identity, and source metadata.
Resume is rejected when dataset, source commit, or effective configuration do
not match. This prevents an MKG-W checkpoint from being used for MKG-Y.

## Helper scripts

- `scripts/build_kaggle_notebook.py`: build the maintained MKG-W notebook.
- `scripts/validate_notebook.py`: validate the generated MKG-W notebook.
- `scripts/build_mkgy_notebook.py`: build MKG-Y from the reusable hardened
  template.
- `scripts/validate_mkgy_notebook.py`: validate MKG-Y configuration, secret
  handling, checkpoint contract, and code-cell syntax.
- `scripts/kaggle_push_run.sh`: build/validate/configure/push MKG-W.
- `scripts/kaggle_mkgy_push.sh`: build/validate/configure/push MKG-Y.
- `scripts/kaggle_monitor_mkgy.sh`: poll MKG-Y status and preserve a local
  monitor log.
- `scripts/kaggle_download_outputs.sh`: download outputs for an explicit
  `owner/kernel` reference.
- `scripts/finalize_momok_mkgy_wandb.py`: finalize safe W&B metadata from
  completed MKG-Y outputs without retraining.
- `scripts/update_momok_main_wandb_report.py`: update the public report from
  verified run metadata; it does not alter run history.

All commands are intended to run from this repository root. No absolute
development-machine path is required.

## Provenance

The public publication branch is based on official MoMoK history at the pinned
commit above. Runtime changes are bounded to compatibility, checkpoint/resume,
validation, and instrumentation support. See
[REPRODUCTION_CHANGES.md](REPRODUCTION_CHANGES.md) and
[TRANSPLANT_MANIFEST.md](publication/TRANSPLANT_MANIFEST.md) for the
publication audit.

