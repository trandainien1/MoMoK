# W&B ablation report render validation

- Production report: [MoMoK — Main Result Reproduction](https://wandb.ai/trandainien1/graphml-momok-reproduction/reports/MoMoK-—-Main-Result-Reproduction--VmlldzoxODA0ODgwMw==)
- Existing report object updated in place: PASS; report ID retained as `VmlldzoxODA0ODgwMw==`.
- Pre-mutation spec backup: `wandb_report/backups/MOMOK_MAIN_REPORT_PRE_ABLATIONS.json`.
- API structure: PASS; seven headings in expected order, six original MKG-W panels retained, four ablation evaluation panels added, cross-dataset and reproducibility sections retained.
- Ablation run selection: PASS; exactly `k2evocuw`, `mkgwngn1`, `mkgwnex1`, `mkgwnrt1` in the ablation PanelGrid.
- Raw `id` query/filter: ABSENT.
- MKG-Y training-loss panel: unchanged at x=`epoch`, y=`train/loss` compared with the pre-update backed-up report spec.
- Anonymous HTTP app shell: PASS, HTTP 200; fetched HTML did not contain Error 1054.
- Temporary API render test: [MoMoK MKG-W Ablation Render Test](https://wandb.ai/trandainien1/graphml-momok-reproduction/reports/MoMoK-MKG-W-Ablation-Render-Test--VmlldzoxODE4Njc0MA==); spec and selected runsets passed.
- Authenticated and anonymous dynamic visual rendering: NOT VERIFIED. No browser or Chrome DevTools runtime is available in this tool environment.
- Human visual review required: open the production report and confirm all panels render and the four-run ablation selector behaves as expected.
