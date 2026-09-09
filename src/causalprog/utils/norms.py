"""Misc collection of norm-like functions for PyTree structures."""

from typing import TypeVar

import jax

PyTree = TypeVar("PyTree")


def l2_normsq(x: PyTree) -> jax.Array:
    """
    Square of the l2-norm of a PyTree.

    This is effectively "sum(elements**2 in leaf for leaf in x)".
    """
    leaves, _ = jax.tree_util.tree_flatten(x)
    return sum(jax.numpy.sum(leaf**2) for leaf in leaves)
