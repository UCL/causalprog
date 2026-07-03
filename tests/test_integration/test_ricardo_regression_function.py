import jax
import jax.numpy as jnp

from causalprog.graph.ricardo import MLPAlias, build_regression_function, example_model
from causalprog.quadrature import UniformWeightMonteCarloGaussianQuadrature as UWMCGQuad

# Note that x64 precision makes a big difference here, since numerical derivatives are
# sensitive to "non-ops". EG Calculations that analytically make no difference to the
# end result, but numerically _can_ affect the output value due to rounding etc.
jax.config.update("jax_enable_x64", val=True)


def test_uy_independent(
    rng_key,
    k_len: int = 5,
    z_len: int = 10,
    n_points: int = 1000,
    n_eval_pts_per_dim: int = 10,
    f_r: MLPAlias = lambda czl, theta_m: theta_m * jnp.ones_like(czl["z"]),
    f_m: MLPAlias = lambda czl, theta_r: czl["c"] * theta_r,
    f_ux: MLPAlias = lambda xzl, theta_x: xzl["x"] * theta_x[0],
    f_y: MLPAlias = lambda xu_y, theta_y: theta_y[0] * jnp.exp(-(xu_y["x"] ** 2)),
    r_analytic: MLPAlias = lambda xzl, theta: theta["theta_y"][0]
    * jnp.exp(-(xzl["x"] ** 2)),
) -> None:
    r"""Build a $U_Y$-independent regression function.

    Under the assumption that $f_Y$ is independent of $U_Y$, the regression function
    $r(x, z, l; \theta)$ should amount to integrating a constant (albeit one that is
    parametrised by the remaining input arguments to $f_Y$).

    Explicitly, in this test we set $f_Y(u_y, x, l; \theta_Y) = \theta_Y[0]e^{-x^2}$.
    The other supporting functions are set to be non-trivial (but also unrealistic) just
    so that we can check the auto-diff isn't making too much of a mess when we take
    derivatives of variables that (in theory) $r$ doesn't depend on.

    Note that increasing the value of `n_eval_pts_per_dim` comes with a massive memory
    and time overhead - this dictates the size of the "grid" over which we are
    evaluating the constructed $r$ and its derivative, and this grid is 7D!

    On x32-only machines, the computed partial derivatives for non-$\theta_Y$ variables
    are known to be of approximately 1e-7 size, which means that `jnp.allclose` will
    flag them as "not being close" to 0. This appears to be a numerical-rounding error,
    since enabling x64-precision calculations makes this issue disappear.
    """

    def f_pi(ucl, theta_pi):
        """Needs to return a `k_len` vector, so must be defined inside the test body."""
        return theta_pi * ucl["c"] + ucl["u_x"] * jnp.arange(k_len)

    g = example_model(
        k=k_len,
        z_len=z_len,
        compute_u_x=f_ux,
        compute_u_y=f_pi,
        compute_phi_x=None,
        compute_x=None,
        compute_y=f_y,
    )
    # Manually attach methods to node for now.FIXME: should be removed once we have
    # a more elegant solution for attaching additional functions to nodes.
    g.get_node("u_y").f_r = f_r
    g.get_node("u_y").f_m = f_m

    given_theta_x = jnp.ones((1,))
    r = build_regression_function(
        g, given_theta_x, UWMCGQuad(n_points, rng_key=rng_key)
    )
    dr = jax.grad(r, argnums=1)

    # Create "grid" across which to evaluate the built functions
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

    assert jnp.allclose(r(xzl, theta), r_analytic(xzl, theta))

    dr_evaluation = dr(xzl, theta)
    for key in theta:
        if key != "theta_y":
            # f_Y has been constructed to only depend on theta_Y
            assert jnp.allclose(dr_evaluation[key], 0.0), (
                f"{key}-derivative not close to 0 (max {dr_evaluation[key].max()})"
            )
        else:
            # theta_y is the "last" key that gets vectorised
            computed_theta_y_deriv = dr_evaluation[key][..., -1]
            # Derivative should be equal to e^{-x^2}.
            analytic_values = jnp.exp(-(xzl["x"] ** 2))
            assert jnp.allclose(computed_theta_y_deriv, analytic_values)
