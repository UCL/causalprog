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
def test_learn_deterministic_fn(
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
