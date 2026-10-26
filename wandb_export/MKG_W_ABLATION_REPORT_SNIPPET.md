# MKG-W Ablation Report Snippet

These are one-seed operational ablations of the completed MoMoK MKG-W main result. W&B tracking mode is `retrospective_import_from_kaggle_logs`; the values below come from each Kaggle log’s terminal legacy `Test set results` summary under `legacy_best_per_metric`. The epoch-2000 evaluation points in the curves are separate observations.

Baseline: **Full Model (Proposed Method)**, run [`k2evocuw`](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/k2evocuw).

| Variant | Run | MRR | Hits@1 | Hits@3 | Hits@10 | Max absolute delta vs Full Model |
|---|---|---:|---:|---:|---:|---:|
| `no_gating_noise` | [`mkgwngn1`](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/mkgwngn1) | 0.3503000000 | 0.2983000000 | 0.3728000000 | 0.4543000000 | 0.0016020590 |
| `no_exid` | [`mkgwnex1`](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/mkgwnex1) | 0.3509000000 | 0.2968000000 | 0.3724000000 | 0.4586000000 | 0.0053945718 |
| `no_relation_temperature` | [`mkgwnrt1`](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/mkgwnrt1) | 0.3586000000 | 0.3059000000 | 0.3759000000 | 0.4628000000 | 0.0095945718 |

## Provenance

| Variant | Kaggle kernel | Version | Completed epoch | Checkpoint SHA256 | Evidence Artifact |
|---|---|---:|---:|---|---|
| `no_gating_noise` | [`nientrandai/momok-mkg-w-no-gating-noise-full`](https://www.kaggle.com/code/nientrandai/momok-mkg-w-no-gating-noise-full) | 1 | 2000/2000 | `58052dd66e83e61a58c1cc9bc5728dcb8285630280c23c77a89bed4e5caa3291` | `momok-mkg-w-no-gating-noise-kaggle-v1-evidence:v0` |
| `no_exid` | [`nientrandai/momok-mkg-w-no-exid-full`](https://www.kaggle.com/code/nientrandai/momok-mkg-w-no-exid-full) | 1 | 2000/2000 | `cf2d30c4883b1aec837da0aad6251ba5dd74b4aaf346d04307ad702cbaa81fc0` | `momok-mkg-w-no-exid-kaggle-v1-evidence:v0` |
| `no_relation_temperature` | [`nientrandai/momok-mkg-w-no-relation-temperature-full`](https://www.kaggle.com/code/nientrandai/momok-mkg-w-no-relation-temperature-full) | 1 | 2000/2000 | `6aa771d6c985fec4ae97225a77c1f24ed40631b6e8e493c8f0607755833d6898` | `momok-mkg-w-no-relation-temperature-kaggle-v1-evidence:v0` |

The three W&B runs each contain 20 observed evaluation records at epochs 100–2000, a baseline comparison table, a string-safe environment table, and a `reproduction-evidence` Artifact. The checkpoint binaries were not uploaded; only their SHA256 hashes are recorded.

These runs use one seed and operational definitions. Their equivalence to the original paper’s Table 2 protocol is not asserted. The report curves show the logged epoch-specific evaluations; the comparison values above are the terminal legacy best-per-metric summary.
