"""Explicit operational definitions for MoMoK ablation variants.

This module is intentionally framework-independent.  It is the contract
between experiment configuration, model/loss integration, checkpoint
metadata, and reporting.  It does not import PyTorch and performs no
training.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Tuple


class VariantConfigError(ValueError):
    """Raised when an ablation configuration is invalid or ambiguous."""


HEADS = ("structure", "image", "text", "joint")
SINGLE_MODALITY_VARIANTS = {
    "structure_only": ("structure",),
    "image_only": ("image",),
    "text_only": ("text",),
}
VARIANT_IDS = frozenset(
    {
        "full",
        *SINGLE_MODALITY_VARIANTS,
        "joint_only",
        "no_relation_temperature",
        "no_gating_noise",
        "no_adaptive_fusion",
        "no_joint_training_loss_only",
        "no_joint_head",
        "no_exid",
        "mu_zero_proxy",
        "k_sweep",
        "mu_sweep",
    }
)


def config_fingerprint(config: dict) -> str:
    """Hash canonical JSON so output/checkpoint namespaces are stable."""

    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VariantConfig:
    """Serializable variant behavior consumed by later runtime adapters."""

    variant_id: str
    active_heads: Tuple[str, ...]
    use_relation_temperature: bool
    use_gating_noise: bool
    use_adaptive_fusion: bool
    fusion_mode: str
    train_joint_head: bool
    infer_joint_head: bool
    use_exid: bool
    exid_mode: str
    n_exp: int
    mu: float

    def to_dict(self) -> dict:
        """Return JSON-compatible configuration metadata."""

        result = asdict(self)
        result["active_heads"] = list(self.active_heads)
        return result


def resolve_variant(variant_id: str, *, n_exp: int, mu: float) -> VariantConfig:
    """Resolve an explicit variant contract without executing model code.

    ``k_sweep`` is the only variant that permits ``n_exp=1``.  In that
    case ExID is disabled explicitly because the current CLUB estimator
    requires two distinct experts.  This prevents silently reporting a
    K=1 run as if it used the normal ExID objective.
    """

    if variant_id not in VARIANT_IDS:
        raise VariantConfigError(f"unknown variant: {variant_id}")
    if not isinstance(n_exp, int) or n_exp < 1:
        raise VariantConfigError("n_exp must be a positive integer")

    if variant_id != "k_sweep" and n_exp < 2:
        raise VariantConfigError(
            f"{variant_id} requires n_exp >= 2 because CLUB samples two experts"
        )

    if variant_id == "k_sweep" and n_exp == 1:
        use_exid = False
        exid_mode = "disabled_k1"
        effective_mu = 0.0
    elif variant_id == "no_exid":
        if mu != 0.0:
            raise VariantConfigError("no_exid requires mu=0")
        use_exid = False
        exid_mode = "disabled"
        effective_mu = 0.0
    elif variant_id == "mu_zero_proxy":
        if mu != 0.0:
            raise VariantConfigError("mu_zero_proxy requires mu=0")
        use_exid = True
        exid_mode = "mu_zero_proxy"
        effective_mu = 0.0
    else:
        use_exid = True
        exid_mode = "active"
        effective_mu = float(mu)

    active_heads = HEADS
    train_joint_head = True
    infer_joint_head = True
    use_relation_temperature = True
    use_gating_noise = True
    use_adaptive_fusion = True
    fusion_mode = "adaptive"

    if variant_id in SINGLE_MODALITY_VARIANTS:
        active_heads = SINGLE_MODALITY_VARIANTS[variant_id]
        train_joint_head = False
        infer_joint_head = False
    elif variant_id == "joint_only":
        active_heads = ("joint",)
        train_joint_head = True
        infer_joint_head = True
    elif variant_id == "no_relation_temperature":
        use_relation_temperature = False
    elif variant_id == "no_gating_noise":
        use_gating_noise = False
    elif variant_id == "no_adaptive_fusion":
        use_adaptive_fusion = False
        fusion_mode = "equal"
    elif variant_id == "no_joint_training_loss_only":
        train_joint_head = False
    elif variant_id == "no_joint_head":
        train_joint_head = False
        infer_joint_head = False
        active_heads = ("structure", "image", "text")

    return VariantConfig(
        variant_id=variant_id,
        active_heads=active_heads,
        use_relation_temperature=use_relation_temperature,
        use_gating_noise=use_gating_noise,
        use_adaptive_fusion=use_adaptive_fusion,
        fusion_mode=fusion_mode,
        train_joint_head=train_joint_head,
        infer_joint_head=infer_joint_head,
        use_exid=use_exid,
        exid_mode=exid_mode,
        n_exp=n_exp,
        mu=effective_mu,
    )
