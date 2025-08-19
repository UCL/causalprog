from collections.abc import Callable

import jax.numpy as jnp
import numpy.typing as npt
import pytest
from numpyro.handlers import condition, do

from causalprog.causal_problem.causal_estimand import (
    HandlerToApply,
    Model,
    _CPComponent,
)
from causalprog.graph import Graph


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


@pytest.fixture
def conditioned_on_x_1(
    two_normal_graph: Callable[..., Graph],
) -> Callable[..., Callable[..., None]]:
    """
    Only intended for use in test_apply_handlers.

    Builds the model expected when we condition on X=1.
    """

    def _inner(**two_normal_graph_options: float) -> Callable[..., None]:
        return condition(
            two_normal_graph(**two_normal_graph_options).model,
            {"X": jnp.atleast_1d(1.0)},
        )

    return _inner


@pytest.fixture
def double_condition(
    two_normal_graph: Callable[..., Graph],
) -> Callable[..., Callable[..., None]]:
    """
    Only intended for use in test_apply_handlers.

    Builds the model expected when we condition on UX=-10, then again on
    UX=10 (which should override the first action).
    """

    def _inner(**two_normal_graph_options: float) -> Callable[..., None]:
        return condition(
            condition(
                two_normal_graph(**two_normal_graph_options).model,
                {"UX": jnp.atleast_1d(-10.0)},
            ),
            {"UX": jnp.atleast_1d(10.0)},
        )

    return _inner


@pytest.fixture
def condition_then_do(
    two_normal_graph: Callable[..., Graph],
) -> Callable[..., Callable[..., None]]:
    """
    Only intended for use in test_apply_handlers.

    Builds the model expected when we first condition on UX=0, and then
    apply do(X = 10). When sampling, we should still draw samples from
    X as per a N(UX, 1.0).
    """

    def _inner(**two_normal_graph_options: float) -> Callable[..., None]:
        return do(
            condition(
                two_normal_graph(**two_normal_graph_options).model,
                {"UX": jnp.atleast_1d(0.0)},
            ),
            {"X": jnp.atleast_1d(10.0)},
        )

    return _inner


@pytest.mark.parametrize(
    ("handlers", "expected_model"),
    [
        pytest.param(
            ((condition, {"X": jnp.atleast_1d(1.0)}),),
            "conditioned_on_x_1",
            id="Condition X to 1",
        ),
        # Should condition on UX=-10, then OVERRIDE this with UX=10.
        pytest.param(
            (
                (condition, {"UX": jnp.atleast_1d(-10.0)}),
                (condition, {"UX": jnp.atleast_1d(10.0)}),
            ),
            "double_condition",
            id="Condition twice on same variable",
        ),
        # Condition UX=0, but then do X=10.
        # Should still observe samples of X given by N(0, 1).
        pytest.param(
            (
                (condition, {"UX": jnp.atleast_1d(0.0)}),
                (do, {"X": jnp.atleast_1d(10.0)}),
            ),
            "condition_then_do",
            id="Condition then do",
        ),
    ],
)
def test_apply_handlers(
    handlers: tuple[HandlerToApply],
    expected_model: Model,
    two_normal_graph: Callable[..., Graph],
    request: pytest.FixtureRequest,
    assert_samples_are_identical,
    run_default_nuts_mcmc,
    two_normal_graph_params: dict[str, float] | None = None,
    do_with_samples: Callable[..., npt.ArrayLike] = lambda **pv: pv["X"].mean(),
) -> None:
    """
    Test that model handlers are correctly applied to graphs.

    Note that the order of the handlers is important, as it dictates
    which effects are applied first.
    """
    if two_normal_graph_params is None:
        two_normal_graph_params = {"mean": 0.0, "cov": 1.0, "cov2": 1.0}
    if isinstance(expected_model, str):
        expected_model = request.getfixturevalue(expected_model)(
            **two_normal_graph_params
        )

    g = two_normal_graph(**two_normal_graph_params)

    cp = _CPComponent(*handlers, do_with_samples=do_with_samples)

    handled_model = cp.apply_effects(g.model)

    handled_mcmc = run_default_nuts_mcmc(handled_model)
    expected_mcmc = run_default_nuts_mcmc(expected_model)

    assert_samples_are_identical(handled_mcmc, expected_mcmc)
