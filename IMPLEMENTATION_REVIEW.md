# Implementation Review

## Publication provenance

- Fork: [trandainien1/MoMoK](https://github.com/trandainien1/MoMoK)
- Upstream: [zjukg/MoMoK](https://github.com/zjukg/MoMoK)
- Pinned upstream commit:
  `99c2df114d48c79708ea0608644b685183d81bd9`
- Publication branch: `course-reproduction`
- Stable snapshot: `course-momok-main-results-v1`

The branch is based directly on official MoMoK history at the pinned commit.
The added files provide Kaggle notebook generation, dataset validation,
bounded runtime compatibility, full-state checkpoint/resume support, and W&B
publication helpers.

## Semantic audit

- Model architecture: unchanged.
- Training objective: unchanged.
- Dataset splits: validated, not rewritten.
- Evaluation semantics: unchanged apart from bounded return-shape
  compatibility.
- Runtime changes: compatibility, validation, checkpoint/resume, and
  instrumentation only.

## Verification gates

- MKG-W notebook build and static validation: PASS.
- MKG-Y notebook build and static validation: PASS.
- Python compilation and shell syntax checks: PASS.
- Official pinned commit ancestry: PASS.
- Public clone/default-branch/tag verification: PASS.
- Credential, checkpoint, and large-file publication scans: PASS.

## Public report

- Report: [MoMoK — Main Result Reproduction](https://wandb.ai/trandainien1/graphml-momok-reproduction/reports/MoMoK-—-Main-Result-Reproduction--VmlldzoxODA0ODgwMw==)
- MKG-W run provenance remains retrospective.
- MKG-Y run provenance remains live Kaggle tracking.
- The report uses no raw `Metric("id")` filter and contains the complete
  clone-to-Kaggle reproduction procedure.

No credentials, checkpoints, embeddings, large outputs, or private local paths
are published in this review.

## MKG-W Ablation W&B Publication

- Publication scope: the completed `no_gating_noise`, `no_exid`, and
  `no_relation_temperature` Kaggle v1 runs; no training was started or resumed.
- Existing Full Model (Proposed Method) baseline: W&B run `k2evocuw`.
- All runs are marked `retrospective_import_from_kaggle_logs` and use
  `legacy_best_per_metric` for their terminal comparison values. The charts
  preserve the 20 observed epoch-specific evaluations from 100 through 2000;
  epoch-2000 metrics are not conflated with the terminal legacy summary.

| Variant | W&B run | MRR | Hits@1 | Hits@3 | Hits@10 | Max absolute delta vs Full Model | Checkpoint SHA256 |
|---|---|---:|---:|---:|---:|---:|---|
| no_gating_noise | [mkgwngn1](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/mkgwngn1) | 0.3503000000 | 0.2983000000 | 0.3728000000 | 0.4543000000 | 0.0016020590 | `58052dd66e83e61a58c1cc9bc5728dcb8285630280c23c77a89bed4e5caa3291` |
| no_exid | [mkgwnex1](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/mkgwnex1) | 0.3509000000 | 0.2968000000 | 0.3724000000 | 0.4586000000 | 0.0053945718 | `cf2d30c4883b1aec837da0aad6251ba5dd74b4aaf346d04307ad702cbaa81fc0` |
| no_relation_temperature | [mkgwnrt1](https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/mkgwnrt1) | 0.3586000000 | 0.3059000000 | 0.3759000000 | 0.4628000000 | 0.0095945718 | `6aa771d6c985fec4ae97225a77c1f24ed40631b6e8e493c8f0607755833d6898` |

Each run has 20 observed training/evaluation records, baseline comparison and
environment tables, and one `ablation-evidence` Artifact. Artifact manifests
were checked to ensure no checkpoint binary was uploaded. The three variants
are described as one-seed operational ablations; exact equivalence to the
paper's Table 2 protocol is not claimed.

The existing public report was updated in place at the same URL and now has
the **Ablation Studies — MKG-W** section. Its one runset selects baseline
`k2evocuw` plus the three ablation runs. No raw ID filter was introduced, the
MKG-Y loss panel is unchanged, and the main baseline history was not modified.
Report API structure and anonymous HTTP app-shell checks pass (HTTP 200; no
Error 1054 in returned HTML). Dynamic authenticated/incognito chart rendering
could not be verified because this session has no browser/Chrome DevTools
runtime; a human visual check remains required.

Publication metadata: `wandb_report/ablation_publication.json`.
Report backup: `wandb_report/backups/MOMOK_MAIN_REPORT_PRE_ABLATIONS.json`.
Report-ready summary: `wandb_export/MKG_W_ABLATION_REPORT_SNIPPET.md`.
