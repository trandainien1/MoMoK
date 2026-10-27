# W&B MKG-W Ablation Publication Design

## Intent

Publish the three completed MKG-W operational ablations as separate retrospective W&B runs and integrate them into the existing public `MoMoK — Main Result Reproduction` report. The report must compare each ablation against the existing MKG-W Full Model (Proposed Method) baseline without changing either main-result run.

## Scope

| Variant | Kaggle kernel | MRR | H@1 | H@3 | H@10 |
|---|---|---:|---:|---:|---:|
| `no_gating_noise` | `nientrandai/momok-mkg-w-no-gating-noise-full` | 0.3503 | 0.2983 | 0.3728 | 0.4543 |
| `no_exid` | `nientrandai/momok-mkg-w-no-exid-full` | 0.3509 | 0.2968 | 0.3724 | 0.4586 |
| `no_relation_temperature` | `nientrandai/momok-mkg-w-no-relation-temperature-full` | 0.3586 | 0.3059 | 0.3759 | 0.4628 |

The baseline is existing W&B run `k2evocuw`, labelled **Full Model (Proposed Method)**. Its final legacy metrics are MRR 0.3514488296, H@1 0.2979644361, H@3 0.3711979410, and H@10 0.4532054282.

## Decisions

- Create one new W&B run per completed ablation under `trandainien1/graphml-momok-reproduction`, group `ablations`, job type `ablation`.
- Mark all three as `tracking_mode=retrospective_import_from_kaggle_logs`; import only observed values from Kaggle logs.
- Create one safe evidence Artifact per variant. Exclude all checkpoint binaries; record filename and SHA256 only.
- Add an **Ablation Studies — MKG-W** section to the existing report with the Full Model baseline row, three variant rows, baseline deltas, real evaluation curves, provenance, and the one-seed/legacy-policy caveat.
- Update the existing report object in place. Keep its safe empty-query run selection and never use raw `id` filters, `Metric("id")`, or SQL-like filters.

## Non-goals

- No retraining, Kaggle version push, checkpoint resume, model change, W&B main-run history change, or MKG-Y visualization change.
- No claim that these one-seed operational variants are an exact reproduction of MoMoK Table 2 before the original protocol is independently verified.

## Post-publication integration

After the W&B/report work is complete and verified, merge both existing
implementation branches into `master` without rewriting or squashing their
history:

1. `course-reproduction`
2. `ablation-implementation`

Perform the integration in a separate worktree created from the current
remote `master`. Preserve both branches and use normal merge commits where a
fast-forward is not possible. Before pushing `master`, prove that the pinned
upstream commit, `course-reproduction`, and `ablation-implementation` are all
ancestors of the integration result. Do not force-push, rebase, reset, delete
the source branches, or resolve a substantive source conflict by discarding
one branch's implementation.

## Privacy

- W&B/Kaggle credentials remain local-only and are excluded from source, artifacts, report text, and logs.
- Safe evidence includes logs, manifests, output inventories, configurations, parsed histories, and checkpoint sidecar hashes.

## Success criteria

- Exactly three new ablation runs have intended configuration, 20 observed evaluation events, final summaries, tables, and evidence Artifacts.
- The report includes the Full Model baseline and all three approved variants.
- The report is publicly renderable without Error 1054 or an access token.
- No credential or checkpoint binary is published.
