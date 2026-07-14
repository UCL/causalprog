from collections.abc import Callable
from typing import Concatenate

import jax
import jax.numpy as jnp
import pytest

from causalprog.utils.norms import PyTree


@pytest.fixture
def jax_enable_x64():
    """Enable x64 precision for a single test.

    Note that x64 precision can make a big difference to results, since numerical
    derivatives are sensitive to "non-ops". EG Calculations that analytically make no
    difference to the end result, but numerically _can_ affect the output value due to
    rounding etc.
    """
    setting = "jax_enable_x64"
    prev_setting_value = jax.config.read(setting)
    jax.config.update(setting, val=True)

    yield

    jax.config.update(setting, prev_setting_value)


@pytest.fixture
def pytree_allclose() -> Callable[Concatenate[PyTree, PyTree, ...], PyTree]:
    """Essentially `jnp.allclose` but allowing for `PyTree` comparison.

    Signature is identical to `jnp.allclose`.
    """

    def _inner(*args, **kwargs):
        return jax.tree_util.tree_all(jax.tree.map(jnp.allclose, *args, **kwargs))

    return _inner
