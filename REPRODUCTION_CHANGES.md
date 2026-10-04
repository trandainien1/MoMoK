# Reproduction Changes

The publication branch is based directly on official MoMoK commit
`99c2df114d48c79708ea0608644b685183d81bd9`.

The added implementation is orchestration and reproducibility tooling:

- Kaggle notebook generation and metadata helpers.
- Exact dataset-scoped feature selection and dataset validation.
- Bounded runtime compatibility patches for the pinned source.
- Full-state checkpoint/resume helpers and validation.
- Safe W&B authentication, live tracking, retrospective publication, and
  report helpers.
- Static notebook validation and focused regression tests.

The notebook applies runtime patches to a checked-out copy of the upstream
source. No permanent upstream model file is modified by this publication
branch.

## Semantic audit

| Area | Changed? | Notes |
|---|---:|---|
| Model architecture | No | No architecture definition is replaced. |
| Training objective | No | The upstream objective and loss calculations remain the source of truth. |
| Dataset splits | No | Dataset files are validated, not rewritten. |
| Evaluation semantics | No | Evaluation compatibility only adapts the upstream return shape. |
| Runtime compatibility | Yes | Boolean-mask and validation-return compatibility for current runtimes. |
| Checkpoint/resume | Yes | Full-state, atomic checkpoint support for Kaggle session recovery. |
| Instrumentation | Yes | Safe runtime diagnostics and optional W&B logging. |
