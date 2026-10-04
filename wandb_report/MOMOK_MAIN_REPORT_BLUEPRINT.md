# MoMoK — Main Result Reproduction

## Runs

- MKG-W: `k2evocuw`, `retrospective_import_from_kaggle_logs`
- MKG-Y: `tp1184w7`, `live_kaggle`
- Source commit: `99c2df114d48c79708ea0608644b685183d81bd9`

## Sections

1. Reproduction Overview
2. Experimental Setup
3. MKG-W Main Result
4. MKG-Y Main Result
5. Cross-Dataset Comparison
6. Reproducibility & Provenance

## Chart policy

All charts use `epoch` as their x-axis. MKG-W includes training loss, MI loss,
MRR, H@1, H@3, and H@10. MKG-Y includes training loss, MRR, H@1, H@3, and
H@10; no MI-loss panel is included because that metric is absent from the
recorded history.

## Provenance

MKG-W was executed on Kaggle and imported into W&B retrospectively from real
logs. MKG-Y was connected to W&B live during Kaggle execution. Both runs
completed 2,000 epochs and are STRONG_MATCH under the group's all-deltas-below-
0.01 criterion. The criterion is not terminology from the original paper.

The MKG-Y Kaggle process returned an error only after training, while creating
an environment table with mixed string/numeric values. That publishing issue
was repaired locally without retraining; the final checkpoint hash is
`db2e47779049d569dda58810b9f8b3aa5e0f4da52b2a9c02677ba399510e7785`.

## Public links

- Report: https://wandb.ai/trandainien1/graphml-momok-reproduction/reports/MoMoK-—-Main-Result-Reproduction--VmlldzoxODA0ODgwMw==
- Project: https://wandb.ai/trandainien1/graphml-momok-reproduction
- MKG-W run: https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/k2evocuw
- MKG-Y run: https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/tp1184w7
- Official implementation: https://github.com/zjukg/MoMoK
- Public reproduction repository: https://github.com/trandainien1/MoMoK
- Stable reproduction tag: https://github.com/trandainien1/MoMoK/tree/course-momok-main-results-v1

## Reproduction procedure

```bash
git clone https://github.com/trandainien1/MoMoK.git
cd MoMoK
git checkout course-momok-main-results-v1
python scripts/build_kaggle_notebook.py
python scripts/validate_notebook.py
bash scripts/kaggle_push_run.sh
```

For MKG-Y, use `build_mkgy_notebook.py`, `validate_mkgy_notebook.py`, and
`kaggle_mkgy_push.sh`. Attach `WANDB_API_KEY` through Kaggle Secrets before
Save & Run All for live W&B tracking. Monitor and download with the maintained
Kaggle helpers. Credentials remain local-only.
