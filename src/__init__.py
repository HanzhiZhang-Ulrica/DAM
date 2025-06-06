"""
DAM: Dynamic Attention Mask for Long-Context LLM Inference Acceleration

This package provides the core implementation of the Dynamic Attention Mask framework.
"""

from .dam_attention import DamLlamaAttention, DamLlamaForCausalLM

__version__ = "0.1.0"
__author__ = "Hanzhi Zhang, Heng Fan, Kewei Sha, Yan Huang, Yunhe Feng"

__all__ = [
    "DamLlamaAttention",
    "DamLlamaForCausalLM",
] 