# W&B Main Result Report Publication

- Report title: MoMoK — Main Result Reproduction
- Report URL: https://wandb.ai/trandainien1/graphml-momok-reproduction/reports/MoMoK-—-Main-Result-Reproduction--VmlldzoxODA0ODgwMw==
- Project URL: https://wandb.ai/trandainien1/graphml-momok-reproduction
- MKG-W run: https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/k2evocuw
- MKG-Y run: https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/tp1184w7
- MKG-W ablation runs:
  - no_gating_noise: https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/mkgwngn1
  - no_exid: https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/mkgwnex1
  - no_relation_temperature: https://wandb.ai/trandainien1/graphml-momok-reproduction/runs/mkgwnrt1
- MKG-W tracking mode: retrospective_import_from_kaggle_logs
- MKG-Y tracking mode: live_kaggle
- MKG-W ablation tracking mode: retrospective_import_from_kaggle_logs
- MKG-W ablation metric policy: legacy_best_per_metric; final table metrics use the terminal legacy summary, while charts preserve epoch-specific evaluation values.
- Source commit: 99c2df114d48c79708ea0608644b685183d81bd9
- MKG-Y checkpoint SHA256: db2e47779049d569dda58810b9f8b3aa5e0f4da52b2a9c02677ba399510e7785
- Report update: PROGRAMMATIC_IN_PLACE_UPDATE
- Raw ID filter: NO
- Error 1054: ABSENT_FROM_UPDATED_SPEC
- HTTP public page: 200
- Browser visual verification: UNAVAILABLE_IN_CURRENT_TOOL_ENVIRONMENT
- Temporary ablation render-test report: https://wandb.ai/trandainien1/graphml-momok-reproduction/reports/MoMoK-MKG-W-Ablation-Render-Test--VmlldzoxODE4Njc0MA==
- Temporary render-test API spec: PASS; baseline/ablation metric panel and table selection checked; raw ID filter: NO
- Public app-shell HTTP status: 200; Error 1054 in fetched app shell: ABSENT
- Dynamic authenticated/anonymous chart rendering: NOT_VERIFIED_BROWSER_UNAVAILABLE
- Public visual check: HUMAN_REVIEW_REQUIRED
- Ablation report runset: `k2evocuw`, `mkgwngn1`, `mkgwnex1`, `mkgwnrt1`
- Ablation panels: `eval/MRR`, `eval/Hits@1`, `eval/Hits@3`, `eval/Hits@10` vs `epoch`
- Ablation table baseline: Full Model (Proposed Method), run `k2evocuw`
- Ablation result rows (legacy terminal summary):

  | Variant | MRR | Hits@1 | Hits@3 | Hits@10 | Max absolute delta vs Full Model |
  |---|---:|---:|---:|---:|---:|
  | Full Model (Proposed Method) | 0.3514488296 | 0.2979644361 | 0.3711979410 | 0.4532054282 | 0 |
  | no_gating_noise | 0.3503000000 | 0.2983000000 | 0.3728000000 | 0.4543000000 | 0.0016020590 |
  | no_exid | 0.3509000000 | 0.2968000000 | 0.3724000000 | 0.4586000000 | 0.0053945718 |
  | no_relation_temperature | 0.3586000000 | 0.3059000000 | 0.3759000000 | 0.4628000000 | 0.0095945718 |
- Pre-update W&B report spec: `wandb_report/backups/MOMOK_MAIN_REPORT_PRE_ABLATIONS.json`
- MKG-Y training-loss panel: unchanged
- MKG-W main run history: unchanged
- No accessToken URL is used.
- Public reproduction repository: https://github.com/trandainien1/MoMoK
- Stable reproduction tag: https://github.com/trandainien1/MoMoK/tree/course-momok-main-results-v1
- Publication branch: `course-reproduction`
- Publication commit: `0be7453e42e34277eb023e2ee3d1e5d3dc7672cf`
- MKG-W run selector context: one `PanelGrid` runset context
- MKG-Y training-loss panel: unchanged
- Reproduction procedure: starts from public Git clone and stable tag
