import jax
import jax.numpy as jnp

from causalprog.graph.ricardo import build_regression_function, example_model
from causalprog.quadrature import UniformWeightMonteCarloGaussianQuadrature as UWMCGQ

# Need to write suitable compute functions!
# Also need to reverse the x -> u_x edge too.
# u_x needs a compute function.
# u_y needs a compute function from c, u_x, and l (?)
# y needs a compute function from u_y and x (and other vars?)
# Nothing else _needs_ a compute function.
# Then the u_y node needs some extra functions tacked onto it...
# and some new nodes added as parents for f_r, f_m etc.


def test_what_am_i_doing(
    rng_key, k_len: int = 5, z_len: int = 10, n_points: int = 10000
) -> None:
    def f_ux(xzl, theta_x):
        return xzl["x"] * theta_x[0]

    def f_pi(ucl, theta_pi):
        return jnp.ones((k_len,))

    def f_y(xu_y, theta_y):
        return theta_y[0] * jnp.exp(-(xu_y["x"] ** 2))

    f_r = lambda *args, **kwargs: jnp.ones((z_len,))
    f_m = lambda *args, **kwargs: -float("inf")

    g = example_model(
        k=k_len,
        z_len=z_len,
        compute_u_x=f_ux,
        compute_u_y=f_pi,
        compute_phi_x=None,
        compute_x=None,
        compute_y=f_y,
    )
    # Manually attach methods to node for now
    g.get_node("u_y").f_r = f_r
    g.get_node("u_y").f_m = f_m

    given_theta_x = jnp.ones((1,))
    r = build_regression_function(g, given_theta_x, UWMCGQ(n_points, rng_key=rng_key))
    dr = jax.grad(r, argnums=1)

    # In this setup, we should have
    # r(x, z, l; theta) = theta_y[0]e^{-x^2} and thus
    # dr(x, z, l; theta)/dtheta = 0 except for the theta_y component, which should be
    # dr(x, z, l; theta)/dtheta_y = e^{-x^2}
    r_analytic = lambda xzl, theta: theta["theta_y"][0] * jnp.exp(-(xzl["x"] ** 2))

    # Create "grid" across which to evaluate the built functions
    n_eval_pts_per_dim = 10
    theta = {
        "theta_m": jnp.linspace(-1.0, 1.0, num=n_eval_pts_per_dim),
        "theta_r": jnp.linspace(-1.0, 1.0, num=n_eval_pts_per_dim),
        "theta_pi": jnp.linspace(-1.0, 1.0, num=n_eval_pts_per_dim),
        "theta_y": jnp.linspace(-1.0, 1.0, num=n_eval_pts_per_dim).reshape(
            n_eval_pts_per_dim, 1
        ),
    }
    xzl = {
        "x": jnp.linspace(-1.0, 1.0, num=n_eval_pts_per_dim),
        "z": jnp.linspace(-1.0, 1.0, num=n_eval_pts_per_dim),
        "l": jnp.linspace(-1.0, 1.0, num=n_eval_pts_per_dim),
    }

    # This "vectorises" our error functions over all inputs.
    # It's hideous, I know, but it is what it is.
    for key in [*xzl.keys(), *theta.keys()]:
        r = jax.vmap(
            r,
            in_axes=(
                {k: None if k != key else 0 for k in xzl},
                {k: None if k != key else 0 for k in theta},
            ),
        )
        r_analytic = jax.vmap(
            r_analytic,
            in_axes=(
                {k: None if k != key else 0 for k in xzl},
                {k: None if k != key else 0 for k in theta},
            ),
        )
        dr = jax.vmap(
            dr,
            in_axes=(
                {k: None if k != key else 0 for k in xzl},
                {k: None if k != key else 0 for k in theta},
            ),
        )

    # Evaluate the vectorised functions to check if they really do match?
    assert jnp.allclose(r(xzl, theta), r_analytic(xzl, theta))
    # Check gradient computations too
    dr_evaluation = dr(xzl, theta)

    for key in theta:
        if key != "theta_y":
            # f_Y has been constructed to only depend on theta_Y
            assert jnp.allclose(dr_evaluation[key], 0.0)
        else:
            # theta_y is the "last" key that gets vectorised
            computed_theta_y_deriv = dr_evaluation[key][..., -1]
            # Derivative should be equal to e^{-x^2}.
            analytic_values = jnp.exp(-(xzl["x"] ** 2))
            assert jnp.allclose(computed_theta_y_deriv, analytic_values)
