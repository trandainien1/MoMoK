"""Pure runtime decisions shared by training, inference, and tests."""

from .variants import VariantConfig


def training_heads(config: VariantConfig) -> tuple[str, ...]:
    """Return prediction heads that contribute supervised training loss."""

    heads = list(config.active_heads)
    if not config.train_joint_head and "joint" in heads:
        heads.remove("joint")
    if config.variant_id == "no_joint_training_loss_only":
        heads = [head for head in heads if head != "joint"]
    return tuple(heads)


def inference_heads(config: VariantConfig) -> tuple[str, ...]:
    """Return prediction heads used for the variant's reported inference."""

    heads = list(config.active_heads)
    if not config.infer_joint_head and "joint" in heads:
        heads.remove("joint")
    return tuple(heads)


def estimator_should_run(config: VariantConfig) -> bool:
    """Whether the CLUB estimator and its optimizer are part of the run."""

    return config.use_exid
