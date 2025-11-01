from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy.typing as npt
import pytest

from causalprog.solvers.iteration_result import IterationResult
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
            jax.tree_util.tree_map(jax.numpy.allclose, result.fn_args, expected)
        )


@pytest.mark.parametrize(
    (
        "history_logging_interval",
        "expected_iters",
    ),
    [
        pytest.param(
            1,
            list(range(11)),
            id="interval=1",
        ),
        pytest.param(
            2,
            list(range(0, 11, 2)),
            id="interval=2",
        ),
        pytest.param(
            3,
            list(range(0, 11, 3)),
            id="interval=3",
        ),
        pytest.param(
            0,
            [],
            id="interval=0 (no logging)",
        ),
        pytest.param(
            -1,
            [],
            id="interval=-1 (no logging)",
        ),
    ],
)
def test_sgd_history_logging_intervals(
    history_logging_interval: int, expected_iters: list[int]
) -> None:
    """Test that history logging intervals work correctly."""

    def obj_fn(x):
        return (x**2).sum()

    initial_guess = jnp.atleast_1d(1.0)

    result = stochastic_gradient_descent(
        obj_fn,
        initial_guess,
        maxiter=10,
        tolerance=0.0,
        history_logging_interval=history_logging_interval,
    )

    # Check that the correct iterations were logged
    assert result.iter_history == expected_iters, (
        f"IterationResult.iter_history logged incorrectly. Expected {expected_iters}."
        f"Got {result.iter_history}"
    )

    # Check that a correct number of fn_args, grad_val, obj_val were logged
    assert len(result.fn_args_history) == len(expected_iters), (
        "IterationResult.fn_args_history logged incorrectly."
        f"Expected {len(expected_iters)} entries. Got {len(result.fn_args_history)}"
    )
    assert len(result.grad_val_history) == len(expected_iters), (
        "IterationResult.grad_val_history logged incorrectly."
        f"Expected {len(expected_iters)} entries. Got {len(result.grad_val_history)}"
    )
    assert len(result.obj_val_history) == len(expected_iters), (
        "IterationResult.obj_val_history logged incorrectly."
        f"Expected {len(expected_iters)} entries. Got {len(result.obj_val_history)}"
    )

    # Check that logged fn_args, grad_val, obj_val line up correctly
    value_and_grad_fn = jax.jit(jax.value_and_grad(obj_fn))

    if len(expected_iters) > 0:
        for fn_args, obj_val, grad_val in zip(
            result.fn_args_history,
            result.obj_val_history,
            result.grad_val_history,
            strict=True,
        ):
            real_obj_val, real_grad_val = value_and_grad_fn(fn_args)

            # Check that logged obj_val and fn_args line up correctly
            assert real_obj_val == obj_val, (
                "Logged obj_val does not match obj_fn evaluated at logged fn_args."
                f"For fn_args {fn_args}, we expected {obj_fn(fn_args)}, got {obj_val}."
            )

            # Check that logged gradient and fn_args line up correctly
            assert real_grad_val == grad_val, (
                "Logged grad_val does not match gradient of obj_fn evaluated at"
                f" logged fn_args. For fn_args {fn_args}, we expected"
                f" {jax.gradient(obj_fn)(fn_args)}, got {grad_val}."
            )


@pytest.mark.parametrize(
    (
        "make_callbacks",
        "expected",
    ),
    [
        (
            lambda cb: cb,
            [0, 1, 2],
        ),
        (
            lambda cb: [cb],
            [0, 1, 2],
        ),
        (
            lambda cb: [cb, cb],
            [0, 0, 1, 1, 2, 2],
        ),
        (
            lambda cb: [],  # noqa: ARG005
            [],
        ),
        (
            lambda cb: None,  # noqa: ARG005
            [],
        ),
    ],
    ids=[
        "single callable",
        "list of one callable",
        "list of two callables",
        "callbacks=[]",
        "callbacks=None",
    ],
)
def test_sgd_callbacks_invocation(
    make_callbacks: Callable, expected: list[int]
) -> None:
    """Test SGD invokes callbacks correctly for all shapes of callbacks input."""

    def obj_fn(x):
        return (x**2).sum()

    calls = []

    def callback(iter_result: IterationResult) -> None:
        calls.append(iter_result.iters)

    callbacks = make_callbacks(callback)

    initial = jnp.atleast_1d(1.0)

    stochastic_gradient_descent(
        obj_fn,
        initial,
        maxiter=2,
        tolerance=0.0,
        callbacks=callbacks,
    )

    assert calls == expected, (
        f"Callback was not called correctly, got {calls}, expected {expected}"
    )


def test_sgd_invalid_callback(raises_context) -> None:
    def obj_fn(x):
        return (x**2).sum()

    initial = jnp.atleast_1d(1.0)

    with raises_context(TypeError("'int' object is not iterable")):
        stochastic_gradient_descent(
            obj_fn,
            initial,
            maxiter=2,
            tolerance=0.0,
            callbacks=42,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "history_logging_interval", [0, 1, 2], ids=lambda v: f"hist:{v}"
)
@pytest.mark.parametrize(
    "make_callbacks",
    [
        lambda cb: cb,
        lambda cb: [cb],
        lambda cb: [cb, cb],
        lambda cb: [],  # noqa: ARG005
        lambda cb: None,  # noqa: ARG005
    ],
    ids=["callable", "list_1", "list_2", "empty", "none"],
)
def test_logging_or_callbacks_affect_sgd_convergence(
    history_logging_interval,
    make_callbacks,
) -> None:
    """Test that logging and callbacks don't affect convergence of SGD solver."""
    calls = []

    def callback(iter_result: IterationResult) -> None:
        calls.append(iter_result.iters)

    callbacks = make_callbacks(callback)

    def obj_fn(x):
        return (x**2).sum()

    initial_guess = jnp.atleast_1d(1.0)

    baseline_result = stochastic_gradient_descent(
        obj_fn,
        initial_guess,
        maxiter=6,
        tolerance=0.0,
        history_logging_interval=0,
    )

    result = stochastic_gradient_descent(
        obj_fn,
        initial_guess,
        maxiter=6,
        tolerance=0.0,
        history_logging_interval=history_logging_interval,
        callbacks=callbacks,
    )

    baseline_attributes = [
        baseline_result.fn_args,
        baseline_result.obj_val,
        baseline_result.grad_val,
        baseline_result.iters,
        baseline_result.successful,
        baseline_result.reason,
    ]

    result_attributes = [
        result.fn_args,
        result.obj_val,
        result.grad_val,
        result.iters,
        result.successful,
        result.reason,
    ]

    for baseline_attr, result_attr in zip(
        baseline_attributes, result_attributes, strict=True
    ):
        assert baseline_attr == result_attr, (
            "Logging or callbacks changed the convergence behaviour of the"
            " solver. For history_logging_interval"
            f" {history_logging_interval}, callbacks {callbacks}, expected"
            f" {baseline_attributes}, got {result_attributes}"
        )
