from collections.abc import Callable, Iterable
from typing import Concatenate

import jax
import jax.numpy as jnp
import pytest

from causalprog.graph.continuous_treatment import MLPAlias
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
def pytree_allclose() -> Callable[Concatenate[PyTree, PyTree, ...], bool]:
    """Essentially `jnp.allclose` but allowing for `PyTree` comparison.

    Signature is identical to `jnp.allclose`.
    """

    def _inner(x: PyTree, y: PyTree, *args, **kwargs):
        return jax.tree_util.tree_all(
            jax.tree.map(lambda xx, yy: jnp.allclose(xx, yy, *args, **kwargs), x, y)
        )

    return _inner


@pytest.fixture
def pytree_all_same_shape() -> Callable[[PyTree, PyTree], bool]:
    """Essentially `x.shape == y.shape`, but allowing for `PyTree` comparison."""

    def _inner(x: PyTree, y: PyTree):
        return jax.tree_util.tree_all(
            jax.tree.map(lambda xx, yy: jnp.shape(xx) == jnp.shape(yy), x, y)
        )

    return _inner


@pytest.fixture
def vectorise_over_dict_args() -> Callable[Concatenate[MLPAlias, ...], MLPAlias]:
    """Vectorise a pure function of dictionary arguments across the dictionary keys.

    This is essentially a wrapper around iterative applications of `jax.vmap` with the
    appropriate `in_axes` specified. The net effect is that if the input `f` was
    being called with a dictionary argument, whose keys were scalar-valued, the returned
    function can be called with the same dictionary argument whose keys are
    vector-valued, and returns a vector-valued output.

    Note that all vmap-ing is done along axis 0. If you want to pass in vector-values
    for some of the dictionary key inputs, ensure that they are aligned along the
    correct axis (each _row_ should be one value of the input).
    """

    def _inner(f: MLPAlias, *dict_keys: Iterable[str]) -> MLPAlias:
        vec_f = f
        all_keys = [key for key_list in dict_keys for key in key_list]
        for key in all_keys:
            vec_f = jax.vmap(
                vec_f,
                in_axes=tuple(
                    {k: None if k != key else 0 for k in arg_keys}
                    for arg_keys in dict_keys
                ),
            )
        return vec_f

    return _inner
