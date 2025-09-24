from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy.typing as npt
import pytest

from causalprog.solvers.sgd import stochastic_gradient_descent
from causalprog.utils.norms import PyTree


@pytest.mark.parametrize(
    (
        "obj_fn",
        "initial_guess",
        "expected",
        "kwargs_to_sgd",
    ),
    [
        pytest.param(
            lambda x: (x**2).sum(),
            jnp.atleast_1d(1.0),
            jnp.atleast_1d(0.0),
            None,
            id="Deterministic x**2",
        ),
        pytest.param(
            lambda x: (x**2).sum(),
            jnp.atleast_1d(10.0),
            "Did not converge after 1 iterations",
            {"maxiter": 1},
            id="Reaches iteration limit",
        ),
        pytest.param(
            lambda x: (x**2).sum(),
            jnp.atleast_1d(1.0),
            jnp.atleast_1d(0.9),
            {
                "convergence_criteria": lambda x, _: jnp.abs(x.sum()),
                "tolerance": 1.0e0,
                "learning_rate": 1e-1,
            },
            id="Converge on function value less than 1",
        ),
        pytest.param(
            lambda x, a: ((x - a) ** 2).sum(),
            jnp.atleast_1d(1.0),
            jnp.atleast_1d(2.0),
            {
                "fn_args": (2.0,),
            },
            id="Fix positional argument",
        ),
        pytest.param(
            lambda x, *, a: ((x - a) ** 2).sum(),
            jnp.atleast_1d(1.0),
            jnp.atleast_1d(2.0),
            {
                "fn_kwargs": {"a": 2.0},
            },
            id="Fix keyword argument",
        ),
    ],
)
def test_sgd(
    obj_fn: Callable[[PyTree], npt.ArrayLike],
    initial_guess: PyTree,
    kwargs_to_sgd: dict[str, Any],
    expected: PyTree | str,
) -> None:
    """Test the SGD method on a (deterministic) problem.

    This is just an assurance check that all the components of the method are working
    as intended. In each test case, we minimise (a variation of) x**2, changing the
    options that we pass to the SGD solver.
    """
    if not kwargs_to_sgd:
        kwargs_to_sgd = {}

    result = stochastic_gradient_descent(obj_fn, initial_guess, **kwargs_to_sgd)

    if isinstance(expected, str):
        assert not result.successful
        assert result.reason == expected
    else:
        assert jax.tree_util.tree_all(
            jax.tree_util.tree_map(jax.numpy.allclose, result.arg_result, expected)
        )
