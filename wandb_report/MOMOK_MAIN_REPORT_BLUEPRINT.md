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
6. Ablation Studies — MKG-W
7. Reproducibility & Provenance

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

## MKG-W ablations

The ablation section uses one runset selecting the MKG-W Full Model baseline
(`k2evocuw`, labelled **Full Model (Proposed Method)**) and the completed
`no_gating_noise` (`mkgwngn1`), `no_exid` (`mkgwnex1`), and
`no_relation_temperature` (`mkgwnrt1`) runs. It includes only observed
`eval/MRR`, `eval/Hits@1`, `eval/Hits@3`, and `eval/Hits@10` curves with
`epoch` on the x-axis.

The table compares each variant with the existing baseline using the terminal
legacy best-per-metric summary. That source is distinct from the epoch-2000
evaluation point shown in the curve. The runs use one seed and are described as
operational ablations; exact equivalence to the paper's Table 2 protocol is
not claimed. All W&B history imports are retrospective from Kaggle logs.

Descriptive observations: `no_gating_noise` is closest to the baseline by
maximum absolute difference (0.0016020590, Hits@3). `no_exid` differs most on
Hits@10 (+0.0053945718) while its MRR is 0.0005488296 below baseline.
`no_relation_temperature` is above baseline on all four terminal legacy
metrics, with its largest difference on Hits@10 (+0.0095945718). These are
one-seed observations and do not establish significance or causality.

| Variant | Run | Artifact | Checkpoint SHA256 |
|---|---|---|---|
| no_gating_noise | `mkgwngn1` | `momok-mkg-w-no-gating-noise-kaggle-v1-evidence:v0` | `58052dd66e83e61a58c1cc9bc5728dcb8285630280c23c77a89bed4e5caa3291` |
| no_exid | `mkgwnex1` | `momok-mkg-w-no-exid-kaggle-v1-evidence:v0` | `cf2d30c4883b1aec837da0aad6251ba5dd74b4aaf346d04307ad702cbaa81fc0` |
| no_relation_temperature | `mkgwnrt1` | `momok-mkg-w-no-relation-temperature-kaggle-v1-evidence:v0` | `6aa771d6c985fec4ae97225a77c1f24ed40631b6e8e493c8f0607755833d6898` |

W&B runs:

- no_gating_noise: https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/mkgwngn1
- no_exid: https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/mkgwnex1
- no_relation_temperature: https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/mkgwnrt1

Baseline comparison values shown in the report (delta is ablation minus the
existing Full Model run):

| Variant | Metric | Full Model | Ablation | Signed Delta | Absolute Delta |
|---|---|---:|---:|---:|---:|
| no_gating_noise | MRR | 0.3514488296 | 0.3503000000 | -0.0011488296 | 0.0011488296 |
| no_gating_noise | Hits@1 | 0.2979644361 | 0.2983000000 | +0.0003355639 | 0.0003355639 |
| no_gating_noise | Hits@3 | 0.3711979410 | 0.3728000000 | +0.0016020590 | 0.0016020590 |
| no_gating_noise | Hits@10 | 0.4532054282 | 0.4543000000 | +0.0010945718 | 0.0010945718 |
| no_exid | MRR | 0.3514488296 | 0.3509000000 | -0.0005488296 | 0.0005488296 |
| no_exid | Hits@1 | 0.2979644361 | 0.2968000000 | -0.0011644361 | 0.0011644361 |
| no_exid | Hits@3 | 0.3711979410 | 0.3724000000 | +0.0012020590 | 0.0012020590 |
| no_exid | Hits@10 | 0.4532054282 | 0.4586000000 | +0.0053945718 | 0.0053945718 |
| no_relation_temperature | MRR | 0.3514488296 | 0.3586000000 | +0.0071511704 | 0.0071511704 |
| no_relation_temperature | Hits@1 | 0.2979644361 | 0.3059000000 | +0.0079355639 | 0.0079355639 |
| no_relation_temperature | Hits@3 | 0.3711979410 | 0.3759000000 | +0.0047020590 | 0.0047020590 |
| no_relation_temperature | Hits@10 | 0.4532054282 | 0.4628000000 | +0.0095945718 | 0.0095945718 |

## Public links

- Report: https://wandb.ai/trandainien1/graphml-momok-reproduction/reports/MoMoK-—-Main-Result-Reproduction--VmlldzoxODA0ODgwMw==
- Project: https://wandb.ai/trandainien1/graphml-momok-reproduction
- MKG-W run: https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/k2evocuw
- MKG-Y run: https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/tp1184w7
- Official implementation: https://github.com/zjukg/MoMoK
- Public reproduction repository: https://github.com/trandainien1/MoMoK
- Stable reproduction tag: https://github.com/trandainien1/MoMoK/tree/course-momok-main-results-v1
- Current course-reproduction branch: https://github.com/trandainien1/MoMoK/tree/course-reproduction
- README with experiment results and reproduce steps: https://github.com/trandainien1/MoMoK/blob/course-reproduction/README.md

## Reproduction procedure

```bash
git clone https://github.com/trandainien1/MoMoK.git
cd MoMoK
git checkout course-reproduction
python scripts/build_kaggle_notebook.py
python scripts/validate_notebook.py
bash scripts/kaggle_push_run.sh
```

For MKG-Y, use `build_mkgy_notebook.py`, `validate_mkgy_notebook.py`, and
`kaggle_mkgy_push.sh`. Attach `WANDB_API_KEY` through Kaggle Secrets before
Save & Run All for live W&B tracking. Monitor and download with the maintained
Kaggle helpers. Credentials remain local-only.

For the completed MKG-W ablations, the README contains the three variant
build/validation commands, Kaggle push and monitoring steps, safe output
recovery, observed-log parsing, and retrospective W&B publication procedure.
