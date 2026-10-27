# W&B MKG-W Ablation Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Publish the three completed MKG-W ablations as independent retrospective W&B runs and add a baseline-centered ablation section to the existing public W&B report.

**Architecture:** A raw-stdout parser produces observed-only exports per variant. A registry-driven publisher creates exactly one deterministic W&B run and safe Artifact per export. The existing report builder is then updated in place to compare baseline run `k2evocuw` against the three new runs using the existing safe runset mechanism.

**Tech Stack:** Python 3, `wandb`, `wandb_workspaces.reports.v2`, Kaggle CLI/API, JSON, CSV, `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-27-wandb-mkgw-ablation-publication-design.md`

## Global Constraints

- Publish only `no_gating_noise`, `no_exid`, and `no_relation_temperature` from completed Kaggle v1 runs.
- Entity/project/group/job type: `trandainien1` / `graphml-momok-reproduction` / `ablations` / `ablation`.
- Baseline is existing run `k2evocuw`, labelled **Full Model (Proposed Method)**; never duplicate or mutate it.
- All histories are retrospective observed-value imports: no interpolation, forward-fill, synthetic points, or history replay.
- Do not upload checkpoints, credentials, caches, or raw large output bundles.
- Do not retrain, push a kernel, alter main metrics/history, or alter the MKG-Y loss panel.
- Do not use `Metric("id")`, `run.id` metric filters, `id =`, or SQL-like run-ID filters.
- Update the existing report referenced by `wandb_report/report_creation.json`; do not create a competing report.

## Review Focus

- Wrong/truncated log or mismatched variant must block publication before W&B mutation.
- Keep epoch-2000 evaluation values in the observed history. The legacy terminal `Test set results` block reports independently best-per-metric values and is the final comparison source, matching the existing MKG-W baseline policy. Never conflate the two sources or require them to be numerically equal.
- `no_exid` must prove `mu=0`, `use_exid=false`, and `exid_mode=disabled`.
- Re-running a publisher cannot duplicate history or reuse a W&B ID for a different config.
- The new four-run report selection must render without Error 1054.

## Planned File Map

| Path | Change |
|---|---|
| `ablations/wandb_publication.py` | Variant registry, immutable provenance, manifest/hash validation, baseline comparisons, evidence allowlist. |
| `scripts/parse_momok_ablation_log.py` | New raw-stdout log parser; existing v4 JSON-log parser remains unchanged. |
| `scripts/download_momok_ablation_outputs.sh` | Credential-neutral downloader for safe small output files only. |
| `scripts/publish_momok_mkgw_ablation_wandb.py` | Idempotent per-variant W&B publisher. |
| `scripts/update_momok_main_wandb_report.py` | In-place ablation report section using safe runsets. |
| `tests/test_ablation_log_parser.py` | Parser final-epoch/no-fabrication tests. |
| `tests/test_ablation_wandb_publication.py` | Registry, table, summary, artifact, and safety tests. |
| `tests/test_ablation_report_spec.py` | Report block/runset/MKG-Y preservation static tests. |
| `wandb_report/*` | Backup, publication metadata, blueprint, and public report summary updates. |

## Task 1: Establish the publication registry

**Files:** Create `ablations/wandb_publication.py`; create `tests/test_ablation_wandb_publication.py`.

**Interfaces produced:**

```python
@dataclass(frozen=True)
class AblationPublication: ...

def publication_for(variant_id: str) -> AblationPublication: ...
def validate_dataset_manifest(manifest: dict, item: AblationPublication) -> None: ...
def comparison_rows(items: Sequence[AblationPublication]) -> list[list[object]]: ...
```

- [ ] Write failing tests for exactly three supported variants, MKG-W fingerprint, source commit, baseline label/run ID, unique run IDs, and expected final values.
- [ ] Run `python -m unittest tests.test_ablation_wandb_publication -v`; confirm failure because the registry is absent.
- [ ] Implement frozen records with deterministic W&B run IDs: `mkgwngn1`, `mkgwnex1`, `mkgwnrt1`. Add final sidecar SHA256, kernel slug, command config, and exact operational definition for each variant.
- [ ] Require `no_exid` fields: `mu=0.0`, `use_exid=False`, `exid_mode="disabled"`.
- [ ] Run focused tests and `python -m unittest discover -s tests -p 'test*.py'`; require PASS.
- [ ] Commit: `feat: define MKG-W ablation publication registry`.

## Task 2: Parse real raw ablation logs

**Files:** Create `scripts/parse_momok_ablation_log.py`; create `tests/test_ablation_log_parser.py`.

**Interfaces produced:**

```python
def parse_raw_ablation_log(path: Path) -> tuple[list[dict], dict, dict]: ...
```

- [ ] Write failing fixture tests for `Epoch N, average loss`, `Epoch: N`, and `test_*` lines; assert parsed epoch/value pairs retain their original epoch.
- [ ] Add tests for repeated evaluation text, absent epoch 2000, and fabricated-point count. The terminal epoch-2000 evaluation must be required for publication.
- [ ] Run `python -m unittest tests.test_ablation_log_parser -v`; confirm failure before implementation.
- [ ] Implement parser output: `history.csv`, `history_summary.json`, `provenance.json`, `audited_final_metrics.json` under a caller-provided `wandb_export/ablations/<variant>/` directory.
- [ ] Parse only observed epoch averages, observed progress-bar MI values when present, and observed evaluation values. Do not infer LR or fill missing cells.
- [ ] Parse final comparison values only from the terminal `Test set results` block; record `metric_policy=legacy_best_per_metric` and its source separately from epoch-2000 eval.
- [ ] Verify all three logs: `last_epoch_found=2000`, `evaluation_points=20`, `fabricated_points=0`, and terminal summary values match the registry to logged precision. Preserve the distinct epoch-2000 evaluation values in history.
- [ ] Commit: `feat: parse observed MKG-W ablation histories`.

## Task 3: Recover safe evidence and validate it

**Files:** Create `scripts/download_momok_ablation_outputs.sh`; modify `.gitignore`; extend `tests/test_ablation_wandb_publication.py`.

- [ ] Write failing tests for the downloader allowlist. It must include only `ablation_smoke.log`, `dataset_manifest.json`, `output_inventory.json`, and `checkpoint_final.pt.sha256`.
- [ ] Implement `download_momok_ablation_outputs.sh <owner/kernel-slug> <destination>`. It assumes an already-authenticated Kaggle CLI and must never source credential files or choose an account.
- [ ] Add generated `kaggle_outputs/ablations/` and `wandb_export/ablations/` paths to `.gitignore`; do not ignore source/config files.
- [ ] Download all three completed outputs and validate manifest fingerprint plus checkpoint sidecar hash against the registry. Never download/commit `.pt` files as part of publishing.
- [ ] Run focused/full tests; require PASS.
- [ ] Commit: `feat: recover safe ablation evidence outputs`.

## Checkpoint A: Evidence gate

- [ ] Each variant has a validated MKG-W manifest, sidecar hash, raw full-training log, 20 observed eval events, and terminal epoch-2000 metrics.
- [ ] No `.pt`, `.pth`, `.env`, `.secrets/`, credential, or large raw log is staged.

## Task 4: Publish one W&B run per ablation

**Files:** Create `scripts/publish_momok_mkgw_ablation_wandb.py`; modify `ablations/wandb_publication.py`; extend `tests/test_ablation_wandb_publication.py`.

**Interfaces produced:**

```python
def publish_variant(variant_id: str, export_dir: Path) -> PublicationResult: ...
```

- [ ] Write fake-adapter tests for config, summary, table schemas, Artifact allowlist, baseline delta math, and idempotency checks.
- [ ] Implement W&B initialization with registry ID and `resume="allow"`; assert returned `run.id` exactly matches. If it already belongs to another config/variant, stop before logging.
- [ ] Define `epoch`, `train/*`, `eval/*`, and `runtime/*` metrics. Log exactly one record per parsed epoch and never log missing fields.
- [ ] Add final summary fields: completed epoch, variant definition, operational-ablation flag, legacy metric policy, final values, baseline signed/absolute deltas, sidecar hash, Kaggle kernel/version, and source commit.
- [ ] Add tables: `ablation_vs_full_model` with baseline first, and string-only `reproduction_environment`.
- [ ] Add individual Artifacts of type `ablation-evidence`: `momok-mkg-w-<variant>-kaggle-v1-evidence`; include only safe export files and metadata, never checkpoint binaries.
- [ ] Publish variants sequentially after silently loading `.secrets/wandb.env`; query W&B API after each to validate run identity, 20 eval points, final epoch, tables, and Artifact.
- [ ] Commit publisher code and safe `wandb_report/ablation_publication.json`: `feat: publish MKG-W ablation evidence to W&B`.

## Task 5: Update the existing public report in place

**Files:** Modify `scripts/update_momok_main_wandb_report.py`, `wandb_report/MOMOK_MAIN_REPORT_BLUEPRINT.md`, `wandb_report/REPORT_PUBLICATION_SUMMARY.md`, `wandb_report/report_creation.json`; create `tests/test_ablation_report_spec.py`.

- [ ] Write failing static tests that require one **Full Model (Proposed Method)** row, all three variants, MRR/H@1/H@3/H@10 plus signed and absolute baseline deltas, and the one-seed/legacy operational-ablation caveat.
- [ ] Assert no raw `id` filter, no Error-1054 pattern, and unchanged MKG-Y training-loss panel declaration.
- [ ] Implement an **Ablation Studies — MKG-W** section after Cross-Dataset Comparison and before Reproducibility & Provenance.
- [ ] Use a runset selecting only baseline `k2evocuw` plus `mkgwngn1`, `mkgwnex1`, and `mkgwnrt1`, with the existing empty-query/run-settings approach. Do not introduce a raw ID query.
- [ ] Add four real comparison panels: eval/MRR, eval/Hits@1, eval/Hits@3, eval/Hits@10. Do not add a synthetic loss panel or a paper-target curve.
- [ ] Add static comparison/provenance table with baseline and three variants, Kaggle links, sidecar hashes, and explicit “operational ablation; one seed; legacy metric policy” language.
- [ ] Back up current report spec to `wandb_report/backups/MOMOK_MAIN_REPORT_PRE_ABLATIONS.json` before mutation.
- [ ] Extend the existing minimal render test to a baseline panel, one ablation panel, and static table. Require Error 1054 absent before production mutation.
- [ ] Update the existing report URL in place with `draft=False`; verify authenticated and public visual rendering. Preserve all main-result blocks and MKG-Y loss panel unchanged.
- [ ] Commit: `docs: add MKG-W ablation evidence to W&B report`.

## Checkpoint B: Publication gate

- [ ] Three new W&B runs have correct ID/config/history/summary/tables/Artifacts.
- [ ] Baseline `k2evocuw` history is untouched.
- [ ] Report has baseline + three variants, real evaluation charts, no raw ID filter, no Error 1054, and public rendering PASS.

## Task 6: Document, audit, and push the ablation branch

**Files:** Modify `IMPLEMENTATION_REVIEW.md`; create `wandb_export/MKG_W_ABLATION_REPORT_SNIPPET.md`; update `wandb_report/REPORT_PUBLICATION_SUMMARY.md`.

- [ ] Record run URLs, Artifact names, Kaggle kernels, final metrics, baseline deltas, checkpoint hashes, and the operational-ablation limitation.
- [ ] Run `python -m unittest discover -s tests -p 'test*.py'` and `python -m py_compile ablations/*.py scripts/parse_momok_ablation_log.py scripts/publish_momok_mkgw_ablation_wandb.py scripts/update_momok_main_wandb_report.py`.
- [ ] Run `git diff --check`, `git status --short`, and a filename/category-only credential scan. Confirm no checkpoint, environment file, secret, or large output is staged.
- [ ] Commit: `docs: record MKG-W ablation publication evidence`.
- [ ] Push: `git push origin ablation-implementation`.

## Task 7: Integrate course and ablation histories into `master`

**Files:** No intended source edits beyond any conflict-resolution files. Use an isolated integration worktree.

**Consumes:** Verified `origin/master`, `origin/course-reproduction`, and the fully published `origin/ablation-implementation` branch.

**Produces:** A normal-history `master` containing both branch histories and all verified implementation commits.

- [ ] Fetch remotes without changing branch heads locally:

```bash
git fetch origin --prune
git fetch upstream --tags
git log --graph --oneline --decorate --all -40
git log --oneline origin/master..origin/course-reproduction
git log --oneline origin/master..origin/ablation-implementation
```

- [ ] Create an integration worktree from `origin/master`, not from either feature branch:

```bash
git worktree add -b integrate-course-and-ablations \
  /home/nien/momok-master-integration origin/master
cd /home/nien/momok-master-integration
```

- [ ] Verify merge bases and source pin before merging. Record the output; stop if the pinned commit is missing or either branch is unrelated to official MoMoK history:

```bash
git cat-file -e 99c2df114d48c79708ea0608644b685183d81bd9^{commit}
git merge-base origin/master origin/course-reproduction
git merge-base origin/master origin/ablation-implementation
```

- [ ] Merge with normal Git merges, preserving all source commit topology:

```bash
git merge --no-ff origin/course-reproduction -m "merge: add course reproduction workflow"
git merge --no-ff origin/ablation-implementation -m "merge: add ablation implementation and W&B publication"
```

Do not use `--squash`, `rebase`, `reset`, `checkout --`, or force-push. If a
conflict affects generated output, credentials, checkpoint paths, or a
semantic source change, stop and present the exact conflict before resolving.
For a documentation-only conflict, retain the public reproduction guide and
ablation/report instructions together, then run the required tests.

- [ ] Verify no branch history was lost:

```bash
git merge-base --is-ancestor 99c2df114d48c79708ea0608644b685183d81bd9 HEAD
git merge-base --is-ancestor origin/course-reproduction HEAD
git merge-base --is-ancestor origin/ablation-implementation HEAD
git rev-list --left-right --count HEAD...origin/course-reproduction
git rev-list --left-right --count HEAD...origin/ablation-implementation
```

Expected: all three ancestor checks PASS; the right-side count is `0` for
each feature branch, demonstrating no commits remain absent from the
integration result.

- [ ] Run the full test/static-validation suite and publication safety audit from the integration worktree. Require clean `git diff --check`, no staged credentials/checkpoints, and no large generated output.

- [ ] Ask for human review of the merge graph and commit-range evidence before pushing `master`. After approval:

```bash
git push origin HEAD:master
```

Do not delete `course-reproduction` or `ablation-implementation` after the
merge; retain them as provenance branches.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Wrong or truncated Kaggle log | Validate kernel, manifest, terminal epoch, and sidecar hash before W&B mutation. |
| Final metrics from wrong epoch | Require explicit epoch 2000 evaluation before publication. |
| W&B ID collision or duplicate history | Query existing run/provenance before logging; fail closed on mismatch. |
| Forge Error 1054 | Keep empty-query runset settings; test temporary render before changing production report. |
| Overstated scientific claim | State operational ablation, one seed, and legacy metric policy in every summary/report comparison. |
| Slow checkpoint binary download | Use only SHA256 sidecar; checkpoint binary stays out of W&B and Git. |

## Dependency graph

```text
Registry + tests
  -> raw-log parser + safe evidence recovery
  -> three W&B retrospective publications
  -> temporary report render test
  -> in-place public report update
  -> public visual verification, documentation, push ablation branch
  -> integration worktree, ancestry proof, human merge review, master push
```

## Self-review

- Tasks cover all approved variants, baseline comparison, evidence publication, in-place report mutation, public verification, and branch push.
- Every behavioral change starts with a focused failing test and ends with verification.
- The plan does not authorize retraining, history mutation, fabricated values, checkpoint upload, credentials, raw-ID filters, MKG-Y chart changes, squash/rebase integration, force-push, or source-branch deletion.
