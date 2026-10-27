# MoMoK Ablation Implementation

## Status

This branch adds a configurable ablation implementation and records three
completed MKG-W operational ablations. The existing MKG-W and MKG-Y Full
Model runs and their histories were not modified.

Implemented infrastructure includes:

- explicit variant configuration and training/evaluation contracts;
- `K=1` handling that disables CLUB/ExID explicitly;
- separate `mu_zero_proxy` and `no_exid` behavior;
- variant-aware model heads, loss, inference, and checkpoints;
- deterministic configuration fingerprints and isolated output paths;
- atomic full-state checkpointing, resume support, and bundle checksums;
- Kaggle smoke notebooks, full notebook generation, and validation;
- observed-only log parsing and retrospective W&B publication helpers.

Completed MKG-W runs are `no_gating_noise`, `no_exid`, and
`no_relation_temperature`, each at 2,000 epochs on one seed. Their results,
baseline comparisons, Kaggle provenance, checkpoint hashes, and W&B links
are recorded in the [README](README.md), the public
[W&B report](https://wandb.ai/trandainien1/graphml-momok-reproduction/reports/MoMoK-—-Main-Result-Reproduction--VmlldzoxODA0ODgwMw==),
and `IMPLEMENTATION_REVIEW.md`.

The ablations are operational variants, not claims of exact equivalence to
every Table 2 protocol detail. The single-seed results are descriptive and do
not estimate variance. Other defined variants and sweeps are implemented as
contracts/configuration surfaces but have not all been run.

## Variant semantics

| Variant | Operational definition | Fidelity label |
|---|---|---|
| `full` | Four heads, adaptive fusion, relation temperature, gating noise, CLUB | baseline anchor |
| `structure_only` | Structure head contributes training loss and inference | paper-inspired pending protocol confirmation |
| `image_only` | Image head contributes training loss and inference | paper-inspired pending protocol confirmation |
| `text_only` | Text head contributes training loss and inference | paper-inspired pending protocol confirmation |
| `joint_only` | Joint head contributes training loss and inference; all three modality inputs remain available to fusion | paper-inspired pending protocol confirmation |
| `no_relation_temperature` | Relation-dependent gate temperature is disabled | operational ablation |
| `no_gating_noise` | Gaussian gate noise is disabled; dropout is unchanged | operational ablation |
| `no_adaptive_fusion` | Learned modality attention is replaced by equal post-projection weights | paper-inspired unless author definition is confirmed |
| `no_joint_training_loss_only` | Joint head is excluded from supervised loss but remains available for inference | explicit diagnostic variant |
| `no_joint_head` | Joint head is excluded from training and inference | explicit diagnostic variant |
| `mu_zero_proxy` | `mu=0`, but CLUB estimator and optimizer still execute | proxy, not faithful ExID removal |
| `no_exid` | CLUB estimator and MI optimizer are not constructed or executed; `mu=0` required | operational ablation |
| `k_sweep` | Expert-count sweep; `K=1` explicitly disables ExID | sweep with K=1 caveat |
| `mu_sweep` | Scalar `mu` sweep with otherwise unchanged Full Model semantics | sweep |

The exact paper protocol for Table 2 is not fully recoverable from the
repository. Results must therefore distinguish `paper_target`,
`paper_inspired`, and `extension`.

## Configuration contract

The framework-independent contract is implemented in:

```text
ablations/variants.py
ablations/runtime_contract.py
ablations/cli.py
ablations/checkpoint_contract.py
```

The implementation does not claim that evaluating one output head from a
Full Model is equivalent to standalone modality training.

## Verification currently available

```bash
python -m unittest discover -s tests -v
python -m py_compile models/RelMoE.py train.py utils/data_loader.py ablations/*.py
```

The repository's lightweight unit suite covers configuration, variant
contracts, checkpoint wiring, publication schemas, and report structure.
Full experiments execute on Kaggle; see `KAGGLE_RECOVERY.md` for recovery
workflow and the README for commands and results.
