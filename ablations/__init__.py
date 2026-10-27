"""Configuration and execution helpers for MoMoK ablation experiments."""

from .variants import VariantConfig, VariantConfigError, resolve_variant

__all__ = ["VariantConfig", "VariantConfigError", "resolve_variant"]
