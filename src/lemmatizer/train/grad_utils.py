"""Shared gradient-tree helpers used by the MLX trainers.

These recurse over nested dict/list grad trees produced by
`mlx.nn.value_and_grad`. A `None` leaf (tied/unused weights) would crash
on `None * s` or `None + x`, so both helpers guard against it explicitly
and let `None` pass through unchanged.
"""

from __future__ import annotations

from typing import Any


def tree_scale(tree: Any, s: float) -> Any:
    """Recursively scale a nested grads tree by ``s``.

    ``None`` leaves pass through unchanged so tied/unused params don't
    crash on ``None * s``.
    """
    if tree is None:
        return None
    if isinstance(tree, dict):
        return {k: tree_scale(v, s) for k, v in tree.items()}
    if isinstance(tree, list):
        return [tree_scale(v, s) for v in tree]
    return tree * s


def tree_add(a: Any, b: Any) -> Any:
    """Element-wise add two nested trees.

    ``None`` on either side passes through: if only one side is non-None,
    that side is returned as-is. This lets an initial ``accumulated = None``
    seed bootstrap the running gradient sum without a special-cased first
    iteration.
    """
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, dict):
        return {k: tree_add(a[k], b[k]) for k in b}
    if isinstance(a, list):
        return [tree_add(a[i], b[i]) for i in range(len(b))]
    return a + b
