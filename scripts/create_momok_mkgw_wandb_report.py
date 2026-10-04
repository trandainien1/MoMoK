#!/usr/bin/env python3
"""Create the report for the completed MoMoK MKG-W Kaggle v4 run.

This is a retrospective documentation publisher.  It never starts a run,
uploads training data, or changes any Kaggle resource.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
from typing import Iterable

import wandb
import wandb_workspaces.reports.v2 as wr


ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "wandb_export"
REPORT_DIR = ROOT / "wandb_report"
PUBLIC_URL_FILE = REPORT_DIR / "public_report_url.txt"
PUBLICATION_SUMMARY = REPORT_DIR / "REPORT_PUBLICATION_SUMMARY.md"
CREATION_METADATA = REPORT_DIR / "report_creation.json"

ENTITY = "trandainien1"
PROJECT = "graphml-momok-reproduction"
RUN_ID = "k2evocuw"
RUN_NAME = "momok-mkgw-main-kaggle-v4-seed10010"
TITLE = "MoMoK — MKG-W Main Result Reproduction"
KAGGLE_URL = "https://www.kaggle.com/code/nientrandai1/momok-mkg-w-reproduction"
SOURCE_URL = "https://github.com/zjukg/MoMoK"
SOURCE_COMMIT = "99c2df114d48c79708ea0608644b685183d81bd9"
PINNED_SOURCE_URL = f"{SOURCE_URL}/tree/{SOURCE_COMMIT}"
CHECKPOINT_SHA256 = "d8d49fc007d111d4f238c206ad6994fb370b36637b8f865f305fcc99517a14c5"

PAPER = {"MRR": 0.3589, "Hits@1": 0.3038, "Hits@3": 0.3754, "Hits@10": 0.4613}
REPRO = {
    "MRR": 0.3514488296,
    "Hits@1": 0.2979644361,
    "Hits@3": 0.3711979410,
    "Hits@10": 0.4532054282,
}
DELTAS = {key: abs(PAPER[key] - REPRO[key]) for key in PAPER}


def _read_json(name: str) -> dict:
    return json.loads((EXPORT / name).read_text())


def _markdown(text: str) -> wr.MarkdownBlock:
    return wr.MarkdownBlock(text=text.strip())


def _setup_table() -> str:
    return """
| Field | Verified value |
|---|---:|
| Dataset | MKG-W |
| Entities | 15,000 |
| Relations | 169 |
| Train triples | 34,196 |
| Validation triples | 4,276 |
| Test triples | 4,274 |
| Image modality | 14,463 available entities; 383-dimensional features |
| Text modality | 14,123 available entities; 384-dimensional features |
| Seed | 10,010 |
| Epochs | 2,000 |
| Batch size | 1,024 |
| Learning rate | 0.001 |
| μ | 0.0001 |
| Embedding dimension | 200 |
| Number of experts | 3 |
| Evaluation frequency | Every 100 epochs |
| Kaggle kernel | [`nientrandai1/momok-mkg-w-reproduction`](%s) |
| Kaggle version | 4 |
| GPU | Tesla T4 |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| Source commit | `%s` |
""" % (KAGGLE_URL, SOURCE_COMMIT)


def _comparison_table() -> str:
    rows = []
    for key in ("MRR", "Hits@1", "Hits@3", "Hits@10"):
        relative = DELTAS[key] / PAPER[key] * 100
        rows.append(
            f"| {key} | {PAPER[key]:.10f} | {REPRO[key]:.10f} | "
            f"{DELTAS[key]:.10f} | {relative:.4f}% |"
        )
    return "\n".join(
        [
            "| Metric | Paper | Reproduced | Absolute Delta | Relative Delta (%) |",
            "|---|---:|---:|---:|---:|",
            *rows,
        ]
    )


def _reproduction_guide() -> str:
    return """
### 5.1 Source and Data Identity

Official repository: [zjukg/MoMoK](%s). The evaluated source is pinned to
`%s`. The dataset fingerprint is MKG-W: 15,000 entities, 169 relations,
34,196/4,276/4,274 train/validation/test triples, image availability
14,463 × 383, and text availability 14,123 × 384. These cardinalities are
validated before training so an incorrect dataset cannot silently be used.

### 5.2 Original Kaggle v4 Execution Provenance

The original execution was
[`nientrandai1/momok-mkg-w-reproduction`](%s), version 4, status COMPLETE,
epoch 2,000/2,000, at source commit `%s`. The notebook set up the runtime,
downloaded official embeddings, validated MKG-W, ran the smoke test, ran the
full training, evaluated the final model, and exported evidence. Its final
checkpoint hash is `%s`. Version 4 was not live-tracked by W&B; this report
uses a retrospective import.

### 5.3 End-to-End Reproduction Procedure

The current checkout has no configured `git remote -v`, so no public clone
URL is asserted here. Use the repository URL supplied by the project owner,
then run the real local commands below from the repository root.

1. **Clone and enter the repository.** Check out the intended reproduction
   commit after cloning; do not invent a remote URL when none is configured
   in the source checkout.
2. **Prepare Kaggle credentials locally.** Copy the real example file with
   `cp .secrets/kaggle.env.example .secrets/kaggle.env`, keep it gitignored,
   and set only the documented variable names. Never put credential values
   in a notebook or report.
3. **Choose W&B tracking mode.** Historical v4 is imported after completion.
   Future live runs should provide `WANDB_API_KEY` through Kaggle Secrets;
   this v4 notebook itself is not represented as live W&B tracking.
4. **Build and validate the notebook:**
   ```bash
   python scripts/build_kaggle_notebook.py
   python scripts/validate_notebook.py
   ```
   These validate notebook JSON, the pinned source, MKG-W configuration,
   paper targets, credential absence, and hardening markers.
5. **Configure and push the Kaggle kernel.** The maintained v4 launcher is
   `bash scripts/kaggle_push_run.sh`; it sources the local Kaggle environment,
   rebuilds and validates the notebook, configures metadata, requests GPU and
   Internet, and pushes the kernel. The v4 execution itself used the kernel
   metadata and launcher state recorded in the local review artifacts.
6. **Notebook stages on Kaggle:** inspect GPU; install dependencies; clone
   MoMoK; checkout `%s`; apply the reviewed compatibility patch; download the
   official embedding archive; validate MKG-W; run the 2-epoch smoke test;
   launch 2,000-epoch training; evaluate at the configured interval; save
   checkpoints; evaluate epoch 2,000; and write the evidence outputs.
7. **Monitor:** source the repository-local Kaggle environment, then use
   `bash scripts/kaggle_monitor_detached.sh` or the status/log commands in
   that wrapper. Do not load credentials into command arguments.
8. **Download outputs:** use `bash scripts/kaggle_download_outputs.sh` with
   the active kernel reference. Expected safe outputs include the manifest,
   environment, smoke summary, full log, final summary/report, and checkpoint
   hash metadata.
9. **Verify the result** against MRR 0.3589, H@1 0.3038, H@3 0.3754, and
   H@10 0.4613. No extra training is used to force a match.
10. **Publish evidence to W&B** using the audited local publisher:
   `python scripts/publish_momok_wandb.py`. It imports only safe logs and
   summaries, creates the comparison/environment tables, and does not upload
   the checkpoint binary by default.

The commands above are the repository's current command surface. The
original v4 execution predates the retrospective W&B publisher; that is a
provenance distinction, not a change to the completed experiment.

### 5.4 Kaggle Notebook Internal Workflow

```text
Local machine
    | build + validate notebook
    v
Kaggle API
    | push kernel
    v
Kaggle GPU runtime
    +--> install dependencies
    +--> clone MoMoK @ pinned commit
    +--> download official embeddings
    +--> validate MKG-W
    +--> smoke test
    +--> 2000-epoch training
    +--> periodic checkpoint
    +--> final evaluation
    v
Kaggle outputs
    +--> logs / metrics / manifest / checkpoint hash
    v
W&B evidence/report
```

### 5.5 Checkpoint and Resume Strategy

The current hardening implementation preserves model state, MI estimator
state, both optimizer states, scheduler state, completed epoch, Python/NumPy/
Torch/CUDA RNG state, corpus/train ordering, and config/source/dataset
metadata. Checkpoints are written atomically with SHA256 sidecars and are
periodic in the current implementation. Model-only state is insufficient for
exact continuation because optimizer, scheduler, RNG, and ordering state also
affect the next update. Version 4's verified evidence is its final checkpoint
artifact; later hardening is documented separately and must not be read back
into the historical execution.

### 5.6 Output Verification

Verification combines dataset identity, source commit, Kaggle kernel/version,
completed epoch, paper-vs-reproduction metrics, and credential-leak checks.
The final checkpoint SHA256 is `%s`. The local checkpoint can be checked with:
`sha256sum kaggle_outputs/version4_checkpoint_final/repro_output/checkpoints/checkpoint_final.pt`.

### 5.7 W&B Tracking Provenance

The W&B run uses `tracking_mode = retrospective_import_from_kaggle_logs`.
The represented history starts at epoch 100 and ends at epoch 2000, with 20
training/evaluation records. Only values actually present in the Kaggle log
were imported; missing intermediate values were not interpolated or
synthesized. Fabricated metrics: NO.
""" % (SOURCE_URL, SOURCE_COMMIT, KAGGLE_URL, SOURCE_COMMIT, CHECKPOINT_SHA256, SOURCE_COMMIT, CHECKPOINT_SHA256)


def _blocks() -> list:
    runset = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="MKG-W v4 — exact run k2evocuw",
        # Use the v2 additive selection tree rather than an id filter.  Forge
        # currently translates raw id filters into invalid SQL (`id` column).
        filters="",
        run_settings={RUN_ID: wr.RunSettings(disabled=False)},
    )
    # The current v2 SDK serializes root=0 selections as the visible run tree.
    # This selects exactly k2evocuw while leaving the query/filter empty.
    runset._selections_root = 0

    panels = [
        wr.LinePlot(title="Training loss", x="epoch", y=["train/loss"], title_x="Epoch"),
        wr.LinePlot(title="MI estimator loss", x="epoch", y=["train/mi_estimator_loss"], title_x="Epoch"),
        wr.LinePlot(title="MRR", x="epoch", y=["eval/MRR"], title_x="Epoch"),
        wr.LinePlot(title="Hits@1", x="epoch", y=["eval/Hits@1"], title_x="Epoch"),
        wr.LinePlot(title="Hits@3", x="epoch", y=["eval/Hits@3"], title_x="Epoch"),
        wr.LinePlot(title="Hits@10", x="epoch", y=["eval/Hits@10"], title_x="Epoch"),
    ]

    return [
        wr.H1("Reproduction Overview"),
        _markdown(
            """
**Paper:** *Multiple Heads Are Better Than One: Mixture of Modality Knowledge
Experts for Entity Representation Learning*<br>
**Venue:** ICLR 2025<br>
**Task:** Multi-Modal Knowledge Graph Completion<br>
**Experiment:** Main-result reproduction<br>
**Dataset:** MKG-W<br>
**Execution platform:** Kaggle GPU<br>
**Training:** 2,000 epochs<br>
**Official source:** [zjukg/MoMoK](%s) at `%s`.

> This W&B run was reconstructed retrospectively from the completed Kaggle v4
> logs and artifacts. The original Kaggle execution itself was not tracked live
> by W&B.
""" % (SOURCE_URL, SOURCE_COMMIT)
        ),
        wr.H1("Experimental Setup"),
        _markdown(_setup_table()),
        _markdown(
            "Safe links: [official source](%s), [pinned source tree](%s), "
            "[Kaggle kernel](%s), [W&B run](https://wandb.ai/%s/%s/runs/%s)."
            % (SOURCE_URL, PINNED_SOURCE_URL, KAGGLE_URL, ENTITY, PROJECT, RUN_ID)
        ),
        wr.H1("Training Process"),
        _markdown(
            "Historical training/evaluation values were reconstructed at the "
            "evaluation checkpoints available in the Kaggle execution log. "
            "The imported history contains 20 represented points at epochs "
            "100 through 2,000; it does not claim 2,000 live W&B points."
        ),
        wr.PanelGrid(runsets=[runset], panels=panels[:2]),
        wr.PanelGrid(runsets=[runset], panels=panels[2:]),
        wr.H1("Paper vs Reproduced Results"),
        _markdown(
            _comparison_table()
            + "\n\nAll four absolute deviations are below 0.01 under the reproduction "
            "criterion defined by our group.\n\n"
            "**Reproduction criterion:** `STRONG_MATCH` means all absolute "
            "deltas ≤ 0.01; `REPRO_MATCH` means all ≤ 0.02; otherwise "
            "`OUTSIDE_EXPECTED_RANGE`. This criterion is defined for this "
            "reproduction study and is not a criterion proposed by the original paper."
        ),
        wr.H1("Reproducibility & Provenance"),
        _markdown(_reproduction_guide()),
    ]


def _minimal_blocks() -> list:
    """Small render diagnostic with the same safe run selection."""
    runset = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="MKG-W v4 render-test — exact run k2evocuw",
        filters="",
        run_settings={RUN_ID: wr.RunSettings(disabled=False)},
    )
    runset._selections_root = 0
    return [
        wr.H1("MoMoK MKG-W Render Test"),
        _markdown("Minimal Forge render diagnostic for run `k2evocuw`."),
        wr.PanelGrid(
            runsets=[runset],
            panels=[wr.LinePlot(title="MRR", x="epoch", y=["eval/MRR"], title_x="Epoch")],
        ),
    ]


def _validate_run() -> dict:
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
    if run.id != RUN_ID or run.name != RUN_NAME:
        raise RuntimeError("target W&B run identity mismatch")
    return {"id": run.id, "name": run.name, "state": run.state, "group": run.group}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimal-test", action="store_true")
    args = parser.parse_args()
    if os.environ.get("WANDB_API_KEY", "").strip() == "":
        raise SystemExit("WANDB_API_KEY is required in the environment")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_info = _validate_run()
    if args.minimal_test:
        report = wr.Report(
            entity=ENTITY,
            project=PROJECT,
            title="MoMoK MKG-W Render Test",
            description="Temporary render diagnostic; safe to delete after validation.",
            blocks=_minimal_blocks(),
            width="fluid",
        )
    else:
        metadata_path = REPORT_DIR / "report_creation.json"
        existing_url = ""
        if metadata_path.exists():
            existing_url = json.loads(metadata_path.read_text()).get("report_url", "")
        if not existing_url:
            raise RuntimeError("existing report URL is required for a non-minimal repair")
        report = wr.Report.from_url(existing_url)
        report.title = TITLE
        report.description = "Audited report for the completed MoMoK MKG-W Kaggle v4 reproduction."
        report.blocks = _blocks()
        report.width = "fluid"
    # Suppress SDK informational logging so stdout remains the final URL only.
    with contextlib.redirect_stdout(io.StringIO()):
        report.save(draft=False)
        share_url = "" if args.minimal_test else report.enable_share_link()

    report_url = report.url
    if args.minimal_test:
        print(report_url)
        return
    PUBLIC_URL_FILE.write_text(share_url + "\n")
    CREATION_METADATA.write_text(
        json.dumps(
            {
                "title": TITLE,
                "report_url": report_url,
                "view_only_url": share_url,
                "entity": ENTITY,
                "project": PROJECT,
                "run_id": RUN_ID,
                "runset_filter": "",
                "selection_root": 0,
                "selected_run_ids": [RUN_ID],
                "run_identity": run_info,
                "panels": ["train/loss", "train/mi_estimator_loss", "eval/MRR", "eval/Hits@1", "eval/Hits@3", "eval/Hits@10"],
                "tracking_mode": "retrospective_import_from_kaggle_logs",
                "render_method": "PROGRAMMATIC_FIXED",
                "authenticated_render": "PENDING_BROWSER_VERIFICATION",
                "anonymous_render": "LOGIN_REQUIRED_IN_HEADLESS_CHECK",
                "error_1054": "RESOLVED_IN_REPORT_SPEC; UI_VERIFICATION_PENDING",
            },
            indent=2,
        )
        + "\n"
    )
    PUBLICATION_SUMMARY.write_text(
        f"# W&B Report Publication\n\n"
        f"- Report title: {TITLE}\n"
        f"- Report URL: {report_url}\n"
        f"- View-only URL: {share_url}\n"
        f"- Run ID: {RUN_ID}\n"
        f"- Kaggle kernel: nientrandai1/momok-mkg-w-reproduction\n"
        f"- Kaggle version: 4\n"
        f"- Source commit: {SOURCE_COMMIT}\n"
        f"- Checkpoint SHA256: {CHECKPOINT_SHA256}\n"
        f"- Tracking mode: retrospective_import_from_kaggle_logs\n"
        f"- Anonymous access verification: pending HTTP/viewer check\n"
    )
    print(report_url)


if __name__ == "__main__":
    main()
