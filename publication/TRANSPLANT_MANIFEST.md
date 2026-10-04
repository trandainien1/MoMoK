# Publication Transplant Manifest

This manifest records the approved transfer from the local reproduction
workspace to the clean publication worktree. The publication branch starts at
the official MoMoK commit `99c2df114d48c79708ea0608644b685183d81bd9`.

## A — Reproduction tooling

- `scripts/build_kaggle_notebook.py`
- `scripts/build_mkgy_notebook.py`
- `scripts/checkpoint_hardening.py`
- `scripts/configure_kaggle_kernel.py`
- `scripts/configure_mkgy_kernel.py`
- `scripts/kaggle_auth.sh`
- `scripts/kaggle_download_outputs.sh`
- `scripts/kaggle_env.sh`
- `scripts/kaggle_mkgy_push.sh`
- `scripts/kaggle_push_run.sh`
- `scripts/kaggle_monitor_detached.sh`
- `scripts/kaggle_monitor_mkgy.sh`
- `scripts/kaggle_retry_checkpointed.sh`
- `scripts/notebook_runtime_hardening.py`
- `scripts/reproduction_config.py`
- `scripts/validate_notebook.py`
- `scripts/validate_mkgy_notebook.py`
- `scripts/wandb_auth.sh`
- `scripts/publish_momok_wandb.py`
- `scripts/finalize_momok_mkgy_wandb.py`
- `scripts/parse_momok_training_log.py`
- `scripts/update_momok_main_wandb_report.py`
- `scripts/create_momok_mkgw_wandb_report.py`
- `scripts/test_kaggle_env.sh`
- `scripts/test_mkgw_reproduction.py`
- `scripts/test_mkgy_reproduction.py`
- `scripts/test_mkgy_finalization.py`
- `repro/configs/mkgy_main.json`
- `repro/state/mkgy_wandb_run_id.txt`
- `kaggle/momok-db15k/` (audited notebook template consumed by the
  reusable builders; it is not an instruction to launch DB15K)
- `kaggle/momok-mkg-y/`

## B — Required upstream compatibility changes

No permanent modifications to official MoMoK source files are transplanted.
The notebook runtime applies bounded compatibility, checkpoint, resume, and
instrumentation changes after checking out the pinned source. The generated
patch and fix metadata are runtime outputs and are not copied into the public
source tree.

## C — Documentation

- `README.md` — replacement reproduction-only README
- `UPSTREAM_README.md` — original official README preserved for attribution
- `REPRODUCTION_CHANGES.md` — scope and semantic-change audit

## D — Generated evidence (excluded)

- `kaggle_outputs/`
- `wandb_export/`
- generated W&B report backups/debug dumps
- training logs, summaries, and downloaded artifacts
- checkpoint files and feature tensors

## E — Sensitive or local-only (excluded)

- `.secrets/`
- `*.env`
- Kaggle and GitHub credentials
- local W&B state/cache
- virtual environments and machine-specific configuration

## F — Unrelated local changes (excluded)

- pre-existing permission/debug helper scripts not required by the public
  reproduction workflow
- local implementation-review state and run-state artifacts
- historical DB15K evidence and private outputs
