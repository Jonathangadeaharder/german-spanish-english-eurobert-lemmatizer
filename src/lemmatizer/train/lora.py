"""Shared LoRA (Low-Rank Adaptation) linear layer for MLX models.

Used by both the multitask EuroBert trainer and the zh BIO trainer to
inject trainable low-rank matrices into frozen base weights.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.weight = base.weight
        self.bias = getattr(base, "bias", None)
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = mx.random.normal((rank, base.weight.shape[1])) * 0.01
        self.lora_b = mx.zeros((base.weight.shape[0], rank))
        self.freeze(keys=["weight"])
        if self.bias is not None:
            self.freeze(keys=["bias"])

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y + ((x @ self.lora_a.T) @ self.lora_b.T) * self.scale
