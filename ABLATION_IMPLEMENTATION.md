# MoMoK Ablation Implementation

## Status

This branch is an isolated implementation workspace for MoMoK ablation
experiments. The completed MKG-W/MKG-Y Full Model artifacts are not modified.

Completed so far:

- explicit variant configuration contract;
- `K=1` handling that disables CLUB/ExID explicitly;
- distinction between `mu_zero_proxy` and `no_exid`;
- head/loss/inference policy contract;
- deterministic configuration fingerprinting;
- optional-estimator checkpoint schema metadata;
- local pure-Python contract tests.

The Kaggle notebook transplant, full-state runtime adapter, and Kaggle smoke
test remain gated until they are implemented and verified.

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

The host environment does not import PyTorch, so GPU/model smoke tests must
run later in a matching Torch/Kaggle environment.
