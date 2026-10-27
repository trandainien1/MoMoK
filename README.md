# MoMoK Reproduction

This public branch is a reproduction workflow built on the official
[zjukg/MoMoK](https://github.com/zjukg/MoMoK) repository. The experiments use
the pinned upstream source commit
`99c2df114d48c79708ea0608644b685183d81bd9`.

The original upstream README is preserved as [UPSTREAM_README.md](UPSTREAM_README.md).
This README documents the reproduction workflow and summarizes the completed
main-result and selected ablation experiments. The detailed curves, run
configuration, and evidence artifacts are linked from the public W&B report.

## Reproduction scope

The maintained workflows cover the MoMoK Full Model main-result reproductions
for MKG-W and MKG-Y and three completed MKG-W operational ablations:
`no_gating_noise`, `no_exid`, and `no_relation_temperature`. The ablation
notebooks and publication helpers are included. This README does not claim
that these one-seed operational variants exactly reproduce every detail of
the paper's Table 2 protocol. PMF and unrelated datasets are outside scope.

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
git checkout course-reproduction
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
ignored local environment file. For local retrospective publishing, create
`.secrets/wandb.env` with the variable names `WANDB_API_KEY`, `WANDB_ENTITY`,
and `WANDB_PROJECT` (entity `trandainien1`, project
`graphml-momok-reproduction`); fill the key locally and never commit or share
the file.

```bash
mkdir -p .secrets
chmod 700 .secrets
${EDITOR:-vi} .secrets/wandb.env
chmod 600 .secrets/wandb.env
```

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

## Completed main-result experiments

Both main-result runs completed 2,000 epochs using source commit
`99c2df114d48c79708ea0608644b685183d81bd9`. The W&B comparison uses the
paper-reported values and the final reproduced metrics. `STRONG_MATCH` is the
group's criterion that every absolute delta is at most 0.01; it is not a term
used by the paper.

| Dataset | Metric | Paper | Reproduced | Absolute delta | W&B run |
|---|---|---:|---:|---:|---|
| MKG-W | MRR | 0.358900 | 0.351449 | 0.007451 | [k2evocuw](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/k2evocuw) |
| MKG-W | Hits@1 | 0.303800 | 0.297964 | 0.005836 | [k2evocuw](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/k2evocuw) |
| MKG-W | Hits@3 | 0.375400 | 0.371198 | 0.004202 | [k2evocuw](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/k2evocuw) |
| MKG-W | Hits@10 | 0.461300 | 0.453205 | 0.008095 | [k2evocuw](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/k2evocuw) |
| MKG-Y | MRR | 0.379100 | 0.377988 | 0.001112 | [tp1184w7](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/tp1184w7) |
| MKG-Y | Hits@1 | 0.350900 | 0.349606 | 0.001294 | [tp1184w7](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/tp1184w7) |
| MKG-Y | Hits@3 | 0.392000 | 0.392978 | 0.000978 | [tp1184w7](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/tp1184w7) |
| MKG-Y | Hits@10 | 0.432000 | 0.427338 | 0.004662 | [tp1184w7](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/tp1184w7) |

Tracking provenance differs by run: MKG-W history was imported retrospectively
from its completed Kaggle logs; MKG-Y was tracked live during Kaggle execution.
The [W&B report](https://wandb.ai/trandainien1/graphml-momok-reproduction/reports/MoMoK-—-Main-Result-Reproduction--VmlldzoxODA0ODgwMw==)
contains the curves, setup, comparison, and provenance details.

## MKG-W ablation experiments

All three variants use MKG-W, seed 10010, 2,000 epochs, evaluation every 100
epochs, batch size 1024, 3 experts, learning rate 0.001, embedding dimension
200, and relation dimension 256. `no_gating_noise` and
`no_relation_temperature` use `mu=0.0001`; `no_exid` uses `mu=0` and disables
the estimator and its optimizer. The completed runs used Kaggle GPU execution
and were published retrospectively from their logs.

The table compares each ablation against the completed **Full Model (Proposed
Method)** baseline. Values are the terminal legacy `Test set results`
summary. Under `legacy_best_per_metric`, the implementation tracks the best
value of each metric independently, so the four values in a row need not come
from one shared epoch or checkpoint. The W&B curves separately show the 20
observed evaluation events from epochs 100 through 2,000; missing points were
not interpolated.

| Variant | MRR | Hits@1 | Hits@3 | Hits@10 | Maximum absolute delta vs Full Model | W&B run |
|---|---:|---:|---:|---:|---:|---|
| **Full Model (Proposed Method)** | **0.3514488296** | **0.2979644361** | **0.3711979410** | **0.4532054282** | — | [k2evocuw](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/k2evocuw) |
| `no_gating_noise` | 0.3503000000 | 0.2983000000 | 0.3728000000 | 0.4543000000 | 0.0016020590 | [mkgwngn1](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/mkgwngn1) |
| `no_exid` | 0.3509000000 | 0.2968000000 | 0.3724000000 | 0.4586000000 | 0.0053945718 | [mkgwnex1](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/mkgwnex1) |
| `no_relation_temperature` | 0.3586000000 | 0.3059000000 | 0.3759000000 | 0.4628000000 | 0.0095945718 | [mkgwnrt1](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/mkgwnrt1) |

### Variant definitions and observations

- `no_gating_noise` disables Gaussian noise in the existing expert-gating
  path; the other configured components remain enabled. Its largest absolute
  difference from the Full Model summary is 0.0016020590 (Hits@3).
- `no_exid` disables the CLUB/ExID estimator and its optimizer and uses
  `mu=0`. Its largest absolute difference is 0.0053945718 (Hits@10); MRR is
  0.0005488296 lower than the Full Model summary.
- `no_relation_temperature` removes relation-conditioned temperature from
  expert gating while keeping relation embeddings in the prediction heads.
  Its four legacy summary metrics are above the Full Model values; the largest
  difference is 0.0095945718 (Hits@10).

These are descriptive, one-seed operational comparisons. The differences do
not establish statistical significance or isolate a causal effect across
seeds. The paper's exact Table 2 implementation protocol is not fully
recoverable from the public source, so these runs are not presented as exact
protocol replications. All three use retrospective W&B imports from completed
Kaggle v1 logs.

| Variant | Kaggle notebook | Checkpoint SHA256 | W&B evidence Artifact |
|---|---|---|---|
| `no_gating_noise` | [Kaggle kernel](https://www.kaggle.com/code/nientrandai/momok-ablation-mkg-w-no-gating-noise-full) | `58052dd66e83e61a58c1cc9bc5728dcb8285630280c23c77a89bed4e5caa3291` | `momok-mkg-w-no-gating-noise-kaggle-v1-evidence:v0` |
| `no_exid` | [Kaggle kernel](https://www.kaggle.com/code/nientrandai/momok-ablation-mkg-w-no-exid-full) | `cf2d30c4883b1aec837da0aad6251ba5dd74b4aaf346d04307ad702cbaa81fc0` | `momok-mkg-w-no-exid-kaggle-v1-evidence:v0` |
| `no_relation_temperature` | [Kaggle kernel](https://www.kaggle.com/code/nientrandai/momok-ablation-mkg-w-no-relation-temperature-full) | `6aa771d6c985fec4ae97225a77c1f24ed40631b6e8e493c8f0607755833d6898` | `momok-mkg-w-no-relation-temperature-kaggle-v1-evidence:v0` |

The checkpoint binaries are not included in the public repository or W&B
evidence artifacts; the recorded SHA256 values identify the verified files.

## Reproducing the MKG-W ablations

The builder creates a self-contained Kaggle notebook pinned to the official
source commit. Use a separate notebook folder for each variant so its
metadata, logs, and Kaggle version remain distinct. The following commands
are the verified builder interface; for a new run, update `id` in that
folder's `kernel-metadata.json` to a kernel slug owned by the authenticated
Kaggle account before pushing.

```bash
# No gating noise
python scripts/build_ablation_kaggle_notebook.py \
  --output kaggle/momok-ablation-full/momok_ablation_full.ipynb \
  --dataset MKG-W --variant no_gating_noise --lr 0.001 --mu 0.0001 \
  --dim 200 --n_exp 3 --epochs 2000 --run_mode full
python scripts/validate_ablation_kaggle_notebook.py \
  kaggle/momok-ablation-full/momok_ablation_full.ipynb

# No ExID (the variant requires mu=0)
python scripts/build_ablation_kaggle_notebook.py \
  --output kaggle/momok-ablation-no-exid-full/momok_ablation_no_exid_full.ipynb \
  --dataset MKG-W --variant no_exid --lr 0.001 --mu 0 \
  --dim 200 --n_exp 3 --epochs 2000 --run_mode full
python scripts/validate_ablation_kaggle_notebook.py \
  kaggle/momok-ablation-no-exid-full/momok_ablation_no_exid_full.ipynb

# No relation temperature
python scripts/build_ablation_kaggle_notebook.py \
  --output kaggle/momok-ablation-no-relation-temperature-full/momok_ablation_no_relation_temperature_full.ipynb \
  --dataset MKG-W --variant no_relation_temperature --lr 0.001 --mu 0.0001 \
  --dim 200 --n_exp 3 --epochs 2000 --run_mode full
python scripts/validate_ablation_kaggle_notebook.py \
  kaggle/momok-ablation-no-relation-temperature-full/momok_ablation_no_relation_temperature_full.ipynb
```

Set up Kaggle authentication locally as described above, then publish and run
each kernel folder with `kaggle kernels push -p <kernel-folder>`. Follow the
run in the Kaggle UI or with `kaggle kernels logs <owner/kernel-slug> --follow`;
check status with `kaggle kernels status <owner/kernel-slug>`. After it
completes, retrieve outputs with `kaggle kernels output <owner/kernel-slug>
-p <local-output-dir>`. Kaggle metadata must use the same owner as the local
Kaggle credentials. Do not launch a run unless the dataset fingerprint and
smoke checks in its generated notebook pass.

For the three already published v1 kernels, the restricted downloader fetches
only safe evidence files and validates the dataset fingerprint and checkpoint
hash:

```bash
bash scripts/download_momok_ablation_outputs.sh \
  nientrandai/momok-mkg-w-no-gating-noise-full \
  kaggle_outputs/ablations/recovered
```

For a new run, retrieve only the allowlisted small files (not the checkpoint
binary), then parse the actual training log:

```bash
OWNER=your-kaggle-username
SLUG=your-kernel-slug
VARIANT=no_gating_noise
EVIDENCE_DIR="kaggle_outputs/ablations/recovered/$SLUG"
mkdir -p "$EVIDENCE_DIR"
kaggle kernels output "$OWNER/$SLUG" -p "$EVIDENCE_DIR" \
  --file-pattern '(^|/)(ablation_smoke\.log|dataset_manifest\.json|output_inventory\.json|checkpoint_final\.pt\.sha256)$'

python scripts/parse_momok_ablation_log.py \
  --log "$EVIDENCE_DIR/ablation_smoke.log" \
  --output-dir "wandb_export/ablations/$VARIANT"
```

Then publish only a registered completed variant. Load `WANDB_API_KEY`,
`WANDB_ENTITY`, and `WANDB_PROJECT` from a local ignored credential file; do
not put their values in shell history, notebook source, or Git:

```bash
set +x
set -a
source .secrets/wandb.env
set +a
VARIANT=no_gating_noise
python scripts/publish_momok_mkgw_ablation_wandb.py \
  --variant "$VARIANT"
```

The publisher validates a fixed registry of the three completed runs and is
not a general publisher for arbitrary reruns. New experiments require a new
run ID and reviewed provenance before publication. The [public W&B project](https://wandb.ai/trandainien1/graphml-momok-reproduction)
and [main report](https://wandb.ai/trandainien1/graphml-momok-reproduction/reports/MoMoK-—-Main-Result-Reproduction--VmlldzoxODA0ODgwMw==)
contain the histories, comparison tables, environment records, and safe
evidence artifacts.

