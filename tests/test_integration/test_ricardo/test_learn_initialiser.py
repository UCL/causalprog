import jax
import jax.numpy as jnp
import pytest

from causalprog.graph.ricardo import learn_initialiser


@pytest.mark.parametrize(
    ("optimiser", "opt_args", "opt_kwargs", "expected_solution"),
    [
        pytest.param(
            None,
            ({"a": 1.5, "b": 0.5, "c": -0.5},),
            {
                "convergence_criteria": lambda x, _: jnp.sqrt(jnp.abs(x)),
                "tolerance": 1e-6,
            },
            {"a": 2.0, "b": 1.0, "c": -1.0},
            id="SDG: Solution basin 1",
        ),
        pytest.param(
            None,
            ({"a": 0.5, "b": 1.5, "c": -0.5},),
            {
                "convergence_criteria": lambda x, _: jnp.sqrt(jnp.abs(x)),
                "tolerance": 1e-6,
            },
            {"a": 1.0, "b": 2.0, "c": -1.0},
            id="SDG: Solution basin 2 (switch initial guess params)",
        ),
    ],
)
def test_learn_initialiser_deterministic_fn(
    optimiser, opt_args, opt_kwargs, expected_solution, pytree_allclose
) -> None:
    """Test that `learn_initialiser` correctly 'learns' the parameters of a
    deterministic function.
    """
    theta_opt = {"a": 1.0, "b": 2.0, "c": -1.0}

    def _r(data, theta):
        return (
            (data["x"] - theta["a"])
            * (data["x"] - theta["b"])
            * (data["x"] - theta["c"])
        )

    eval_pts = {"x": jnp.linspace(-5.0, 5.0, num=25)}
    r_hat_pts = _r(eval_pts, theta_opt)

    result = learn_initialiser(
        _r,
        eval_pts,
        r_hat_pts,
        optimiser=optimiser,
        optimiser_args=opt_args,
        optimiser_kwargs=opt_kwargs,
    )

    assert result.successful
    assert pytree_allclose(result.fn_args, expected_solution)


def _n_eval() -> int:
    """Magic number function for the number of evaluation points in tests.

    `pytest` fixtures cannot be evaluated as part of a parameter value. As such,
    in order to avoid magic numbers everywhere, we have a hidden function to
    fix a value for the number of evaluation points to use.
    """
    return 5


@pytest.mark.parametrize(
    (
        "eval_pts",
        "eval_pts_mapping",
        "r_hat_pts",
        "expected_solution",
    ),
    [
        pytest.param(
            {"x": jnp.linspace(-1.0, 1.0, num=_n_eval())},
            None,
            jnp.zeros(_n_eval()),
            {"a": 0.0, "b": 0.0},
            id="Standard scalar inputs.",
        ),
        pytest.param(
            {"x": jnp.tile(jnp.linspace(-1.0, 1.0, num=_n_eval()), (3, 1))},
            {"x": 1},
            jnp.ones(_n_eval()),
            {"a": 1.0, "b": jnp.zeros(3)},
            id="Vector-valued data",
        ),
        pytest.param(
            {"x": jnp.tile(jnp.linspace(-1.0, 1.0, num=_n_eval()), (3, 1)).T},
            {"x": 0},
            jnp.ones(_n_eval()),
            {"a": 1.0, "b": jnp.zeros(3)},
            id="Vector-valued data, transposed",
        ),
        pytest.param(
            {"x": jnp.tile(jnp.linspace(-1.0, 1.0, num=_n_eval()), (3, 4, 1))},
            {"x": 2},
            jnp.ones(_n_eval()),
            {"a": 1.0, "b": jnp.zeros((3, 4))},
            id="Matrix-valued data",
        ),
        pytest.param(
            {"x": jnp.tile(jnp.linspace(-1.0, 1.0, num=_n_eval()), (3, 1))},
            {"x": 1},
            jnp.array([6.0, 9.0 / 4.0, 1.0, 9.0 / 4.0, 6.0]),
            {"a": 1.0, "b": jnp.array([2.0, 1.0, 0.0])},
            id="Non-trivial problem",
        ),
    ],
)
def test_learn_initialiser_evaluation_points_axes_mapping(
    eval_pts: dict[str, jax.Array],
    eval_pts_mapping: dict[str, int],
    r_hat_pts: jax.Array,
    expected_solution: dict[str, jax.Array | float],
    pytree_allclose,
    pytree_all_same_shape,
) -> None:
    """Check that `evaluation_points_axes_mapping` is respected.

    This is tested by passing in data of various sizes, and confirming that the
    optimisation still runs and the resulting output has the expected shape for
    the $\theta$ parameters.
    """

    def _r(data, theta):
        return ((theta["b"] * data["x"]) ** 2).sum() + theta["a"]

    # Since we're only checking array dimension matching,
    # start the solver at the solution to immediately terminate.
    result = learn_initialiser(
        _r,
        eval_pts,
        r_hat_pts,
        evaluation_points_axes_mapping=eval_pts_mapping,
        optimiser_args=(expected_solution,),
    )

    assert result.successful
    assert pytree_all_same_shape(result.fn_args, expected_solution)
    assert pytree_allclose(result.fn_args, expected_solution)
