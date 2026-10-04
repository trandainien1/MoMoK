#!/usr/bin/env python3
"""Integrate the completed MKG-Y result into the existing public report."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

import wandb
import wandb_workspaces.reports.v2 as wr


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "wandb_report"
META = REPORT_DIR / "report_creation.json"
PUBLIC_URL = REPORT_DIR / "public_report_url.txt"
ENTITY = "trandainien1"
PROJECT = "graphml-momok-reproduction"
TITLE = "MoMoK — Main Result Reproduction"
SOURCE = "https://github.com/zjukg/MoMoK"
PUBLIC_REPO = "https://github.com/trandainien1/MoMoK"
PUBLIC_TAG = "https://github.com/trandainien1/MoMoK/tree/course-momok-main-results-v1"
COMMIT = "99c2df114d48c79708ea0608644b685183d81bd9"
MKGW = "k2evocuw"
MKGY = "tp1184w7"


def markdown(text: str) -> wr.MarkdownBlock:
    return wr.MarkdownBlock(text=text.strip())


def runset(run_id: str, name: str) -> wr.Runset:
    # Supported report selection: empty query + disabled settings for all
    # non-target runs, with the visible selection tree rooted at zero.
    selected = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name=name,
        filters="",
        run_settings={run_id: wr.RunSettings(disabled=False)},
    )
    selected._selections_root = 0
    return selected


def comparison_table() -> str:
    rows = [
        ("MKG-W MRR", 0.3589, 0.3514488296, 0.0074511704),
        ("MKG-W H@1", 0.3038, 0.2979644361, 0.0058355639),
        ("MKG-W H@3", 0.3754, 0.3711979410, 0.0042020590),
        ("MKG-W H@10", 0.4613, 0.4532054282, 0.0080945718),
        ("MKG-Y MRR", 0.3791, 0.3779882618405216, 0.0011117381594784),
        ("MKG-Y H@1", 0.3509, 0.34960570784829137, 0.0012942921517086),
        ("MKG-Y H@3", 0.3920, 0.3929778445362373, 0.0009778445362373),
        ("MKG-Y H@10", 0.4320, 0.42733758918512954, 0.0046624108148705),
    ]
    return "\n".join([
        "| Dataset / Metric | Paper | Reproduced | Absolute Delta |",
        "|---|---:|---:|---:|",
        *[f"| {m} | {p:.10f} | {r:.10f} | {d:.10f} |" for m, p, r, d in rows],
    ])


def blocks() -> list:
    mkgw_set = runset(MKGW, "MKG-W — exact run k2evocuw")
    mkgy_set = runset(MKGY, "MKG-Y — exact run tp1184w7")
    mkgy_panels = [
        wr.LinePlot(title="MKG-Y training loss", x="epoch", y=["train/loss"], title_x="Epoch"),
        wr.LinePlot(title="MKG-Y MRR", x="epoch", y=["eval/MRR"], title_x="Epoch"),
        wr.LinePlot(title="MKG-Y Hits@1", x="epoch", y=["eval/Hits@1"], title_x="Epoch"),
        wr.LinePlot(title="MKG-Y Hits@3", x="epoch", y=["eval/Hits@3"], title_x="Epoch"),
        wr.LinePlot(title="MKG-Y Hits@10", x="epoch", y=["eval/Hits@10"], title_x="Epoch"),
    ]
    mkgw_panels = [
        wr.LinePlot(title="MKG-W training loss", x="epoch", y=["train/loss"], title_x="Epoch"),
        wr.LinePlot(title="MKG-W MI estimator loss", x="epoch", y=["train/mi_estimator_loss"], title_x="Epoch"),
        wr.LinePlot(title="MKG-W MRR", x="epoch", y=["eval/MRR"], title_x="Epoch"),
        wr.LinePlot(title="MKG-W Hits@1", x="epoch", y=["eval/Hits@1"], title_x="Epoch"),
        wr.LinePlot(title="MKG-W Hits@3", x="epoch", y=["eval/Hits@3"], title_x="Epoch"),
        wr.LinePlot(title="MKG-W Hits@10", x="epoch", y=["eval/Hits@10"], title_x="Epoch"),
    ]
    return [
        wr.H1("Reproduction Overview"),
        markdown(f"""
**Paper:** *Multiple Heads Are Better Than One: Mixture of Modality Knowledge Experts for Entity Representation Learning*
**Task:** Multimodal knowledge graph link prediction
**Experiments:** MKG-W and MKG-Y main-result reproductions, both completed at 2,000 epochs.
**Source:** [zjukg/MoMoK]({SOURCE}) at `{COMMIT}`.

Both runs satisfy the group's **STRONG_MATCH** criterion: all reported
absolute metric deltas are below 0.01. This is our reproduction criterion,
not terminology proposed by the MoMoK paper.

Runs: [MKG-W](https://wandb.ai/{ENTITY}/{PROJECT}/runs/{MKGW}) · [MKG-Y](https://wandb.ai/{ENTITY}/{PROJECT}/runs/{MKGY})
"""),
        wr.H1("Experimental Setup"),
        markdown(f"""
**Official implementation:** [zjukg/MoMoK]({SOURCE})

**Public reproduction repository:** [{PUBLIC_REPO}]({PUBLIC_REPO})

**Pinned upstream commit:** [`{COMMIT}`]({SOURCE}/commit/{COMMIT})

**Stable reproduction tag:** [{PUBLIC_TAG}]({PUBLIC_TAG})

| Setting | MKG-W | MKG-Y |
|---|---:|---:|
| Epochs | 2000 | 2000 |
| Batch size | 1024 | 1024 |
| Seed | 10010 | 10010 |
| Experts | 3 | 3 |
| Evaluation | every 100 epochs | every 100 epochs |
| Learning rate | 0.001 | 0.0005 |
| μ | 0.0001 | 0.0001 |
| Embedding dimension | 200 | 300 |
| Entities | 15000 | 15000 |
| Relations | 169 | 28 |
| Train triples | 34196 | 21310 |
| Valid triples | 4276 | 2665 |
| Test triples | 4274 | 2663 |
| Execution | Kaggle GPU | Kaggle / Tesla T4 |
"""),
        wr.H1("MKG-W Main Result"),
        markdown("Tracking mode: `retrospective_import_from_kaggle_logs`. The Kaggle execution completed first; W&B history was imported later from real logs. No live-tracking claim is made."),
        wr.PanelGrid(runsets=[mkgw_set], panels=mkgw_panels),
        markdown("""
| Metric | Paper | Reproduced | Absolute Delta |
|---|---:|---:|---:|
| MRR | 0.358900 | 0.351449 | 0.007451 |
| Hits@1 | 0.303800 | 0.297964 | 0.005836 |
| Hits@3 | 0.375400 | 0.371198 | 0.004202 |
| Hits@10 | 0.461300 | 0.453205 | 0.008095 |

Verdict: **STRONG_MATCH**.
"""),
        wr.H1("MKG-Y Main Result"),
        markdown("""
Tracking mode: `live_kaggle`. The run contains 2,000 unique training epochs
and 20 evaluation events. The recorded history has no `train/mi_estimator_loss`
field, so no MI-loss panel is fabricated or reconstructed.
"""),
        wr.PanelGrid(runsets=[mkgy_set], panels=mkgy_panels[:1]),
        wr.PanelGrid(runsets=[mkgy_set], panels=mkgy_panels[1:]),
        markdown("""
| Metric | Paper | Reproduced | Absolute Delta |
|---|---:|---:|---:|
| MRR | 0.379100 | 0.377988 | 0.001112 |
| Hits@1 | 0.350900 | 0.349606 | 0.001294 |
| Hits@3 | 0.392000 | 0.392978 | 0.000978 |
| Hits@10 | 0.432000 | 0.427338 | 0.004662 |

Verdict: **STRONG_MATCH**. The run used a Tesla T4 for approximately 6h41m.
The completed epoch-2000 checkpoint hash is
`db2e47779049d569dda58810b9f8b3aa5e0f4da52b2a9c02677ba399510e7785` and the
evidence artifact is `momok-mkg-y-kaggle-evidence:v0`. The Kaggle process
returned an error only after training, while constructing a W&B environment
table with mixed string/numeric values; this was repaired without retraining.
"""),
        wr.H1("Cross-Dataset Comparison"),
        markdown(comparison_table() + """

Maximum absolute delta: MKG-W `0.0080945718`; MKG-Y `0.004662`. Both are
`STRONG_MATCH`. These deltas are a reproducibility comparison, not a claim
that one dataset/model run is statistically better than the other.
"""),
        wr.H1("Reproducibility & Provenance"),
        markdown(f"""
### 6.1 Source identity

Official repository: [zjukg/MoMoK]({SOURCE}); pinned commit `{COMMIT}`.

### 6.2 Dataset identity

MKG-W was validated as 15,000 entities, 169 relations, and 34,196/4,276/4,274
train/validation/test triples, with image availability 14,463 × 383 and text
availability 14,123 × 384. MKG-Y was validated as 15,000 entities, 28
relations, and 21,310/2,665/2,663 triples, with feature tensors `[15000, 383]`
and `[15000, 384]`.

### 6.3 Kaggle execution

Both experiments ran on Kaggle GPUs for 2,000 epochs. MKG-Y used a Tesla T4
and ran for approximately 6h41m.

### 6.4 W&B tracking provenance

MKG-W uses `retrospective_import_from_kaggle_logs`; MKG-Y uses `live_kaggle`.
These modes are intentionally distinguished.

### 6.5 Checkpoint strategy

The hardened workflow supports periodic full-state checkpoints containing model,
MI estimator, both optimizers, scheduler, RNG, corpus ordering, and provenance
metadata. MKG-Y's final checkpoint is epoch 2000 with SHA256
`db2e47779049d569dda58810b9f8b3aa5e0f4da52b2a9c02677ba399510e7785`. The
binary is not uploaded to the W&B artifact; only its hash and provenance are.

### 6.6 Reproduction procedure

**Public reproduction repository:** [{PUBLIC_REPO}]({PUBLIC_REPO})

```bash
git clone {PUBLIC_REPO}.git
cd MoMoK
git checkout course-momok-main-results-v1
```

Then configure the local Python/Kaggle environment, create the ignored
Kaggle credential file, and source `scripts/kaggle_env.sh`. Build and validate
the selected notebook with:

```bash
python scripts/build_kaggle_notebook.py
python scripts/validate_notebook.py
```

For MKG-W, the maintained launcher is `bash scripts/kaggle_push_run.sh`. For
MKG-Y, use `python scripts/build_mkgy_notebook.py`,
`python scripts/validate_mkgy_notebook.py`, and
`bash scripts/kaggle_mkgy_push.sh`. After uploading the MKG-Y notebook,
attach `WANDB_API_KEY` through Kaggle Add-ons → Secrets, enable notebook
access, and select Save & Run All. This manual gate is required for live W&B
tracking and is not replaced by a credential in the repository.

Monitor with `bash scripts/kaggle_monitor_mkgy.sh` and download outputs with
`bash scripts/kaggle_download_outputs.sh owner/kernel`. The workflow validates
the pinned source and dataset before smoke testing, trains the configured
full-model run, writes full-state checkpoints, evaluates the final epoch, and
exports safe evidence. Credentials remain local-only; no raw credential or
access-token URL is part of this report.
"""),
    ]


def validate_run(run_id: str) -> dict:
    run = wandb.Api().run(f"{ENTITY}/{PROJECT}/{run_id}")
    if run.id != run_id:
        raise RuntimeError(f"run identity mismatch: {run.id}")
    return {"id": run.id, "name": run.name, "state": run.state, "group": run.group}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimal-test", action="store_true")
    parser.add_argument("--delete-temp-url", default="")
    args = parser.parse_args()
    if not os.environ.get("WANDB_API_KEY", "").strip():
        raise SystemExit("WANDB_API_KEY is required in the environment")
    validate_run(MKGW)
    validate_run(MKGY)
    if args.minimal_test:
        report = wr.Report(
            entity=ENTITY, project=PROJECT, title="MoMoK Main Render Test",
            description="Temporary two-run render diagnostic.",
            blocks=[wr.H1("Render Test"), markdown("MKG-W and MKG-Y selection test."),
                    wr.PanelGrid(runsets=[runset(MKGW, "MKG-W")], panels=[wr.LinePlot(title="MKG-W MRR", x="epoch", y=["eval/MRR"], title_x="Epoch")]),
                    wr.PanelGrid(runsets=[runset(MKGY, "MKG-Y")], panels=[wr.LinePlot(title="MKG-Y MRR", x="epoch", y=["eval/MRR"], title_x="Epoch")]),
                    markdown("| Dataset | Metric |\n|---|---|\n| MKG-W | MRR |\n| MKG-Y | MRR |")],
            width="fluid",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            report.save(draft=False)
        print(report.url)
        return
    metadata = json.loads(META.read_text())
    report = wr.Report.from_url(metadata["report_url"])
    report.title = TITLE
    report.description = "Public main-result evidence for the completed MoMoK MKG-W and MKG-Y reproductions."
    report.blocks = blocks()
    report.width = "fluid"
    with contextlib.redirect_stdout(io.StringIO()):
        report.save(draft=False)
    PUBLIC_URL.write_text(report.url + "\n")
    META.write_text(json.dumps({
        "title": TITLE, "report_id": report.id, "report_url": report.url,
        "entity": ENTITY, "project": PROJECT, "selected_run_ids": [MKGW, MKGY],
        "selection_method": "empty_query_plus_run_settings_selection_root_0",
        "raw_id_filter_used": False, "render_method": "PROGRAMMATIC_IN_PLACE_UPDATE",
        "panels_mkgw": ["train/loss", "train/mi_estimator_loss", "eval/MRR", "eval/Hits@1", "eval/Hits@3", "eval/Hits@10"],
        "panels_mkgy": ["train/loss", "eval/MRR", "eval/Hits@1", "eval/Hits@3", "eval/Hits@10"],
        "mkgw_tracking_mode": "retrospective_import_from_kaggle_logs",
        "mkgy_tracking_mode": "live_kaggle",
        "error_1054": "NOT_PRESENT_IN_SPEC",
    }, indent=2, ensure_ascii=False) + "\n")
    print(report.url)


if __name__ == "__main__":
    main()
