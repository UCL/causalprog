import jax.numpy as jnp

from causalprog.graph.ricardo import learn_initialiser


def test_optimise_regression_fn(pytree_allclose) -> None:
    theta_opt = {"a": 1.0, "b": 2.0, "c": -1.0}
    theta_0 = {"a": 0.5, "b": 1.5, "c": -0.5}

    r = lambda data, theta: (
        (data["x"] - theta["a"]) * (data["x"] - theta["b"]) * (data["x"] - theta["c"])
    )
    r_hat = lambda data: r(data, theta_opt)

    eval_pts = {"x": jnp.linspace(-5.0, 5.0, num=50)}
    r_hat_pts = r_hat(eval_pts)

    result = learn_initialiser(
        r,
        eval_pts,
        r_hat_pts,
        optimiser_args=(theta_0,),
        convergence_criteria=lambda x, _: jnp.sqrt(jnp.abs(x)),
        tolerance=1e-6,
    )

    assert result.successful
    assert pytree_allclose(result.fn_args, theta_opt)
