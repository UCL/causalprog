from collections.abc import Callable

import jax.numpy as jnp
import numpy.typing as npt
import pytest

from causalprog.causal_problem.causal_estimand import _CPComponent


@pytest.mark.parametrize(
    ("expression", "samples", "expect_error"),
    [
        pytest.param(
            lambda **pv: jnp.atleast_1d(0.0), {}, None, id="Constant expression"
        ),
        pytest.param(
            lambda **pv: jnp.atleast_1d(0.0),
            {"not_needed": jnp.atleast_1d(0.0)},
            None,
            id="Un-needed samples",
        ),
        pytest.param(
            lambda **pv: pv["a"],
            {"a": jnp.atleast_1d(1.0)},
            None,
            id="All needed samples given",
        ),
        pytest.param(
            lambda **pv: pv["b"],
            {"a": jnp.atleast_1d(1.0)},
            KeyError("b"),
            id="Missing sample",
        ),
    ],
)
def test_call(
    expression: Callable,
    samples: dict[str, npt.ArrayLike],
    expect_error: Exception | None,
    raises_context,
) -> None:
    """Check that _CPComponent correctly calls its _do_with_samples attribute."""

    component = _CPComponent(do_with_samples=expression)

    assert callable(component)

    if expect_error:
        with raises_context(expect_error):
            component(samples)
    else:
        assert jnp.allclose(component(samples), expression(**samples))
