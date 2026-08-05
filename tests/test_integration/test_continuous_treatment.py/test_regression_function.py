import jax
import jax.numpy as jnp
import numpy as np

from causalprog._types import MLPAlias


def test_fy_independent_of_uy(
    jax_enable_x64,  # noqa: ARG001
    ricardo_regression_function,
    vectorise_over_dict_args,
    uy_independent_mlps,
    k_len: int = 5,
    z_len: int = 10,
    n_points: int = 1000,
    n_eval_pts_per_dim: int = 10,
) -> None:
    r"""Under the assumption that $f_Y$ is independent of $U_Y$, the regression function
    $r(x, z, l; \theta)$ should amount to integrating a constant (albeit one that is
    parametrised by the remaining input arguments to $f_Y$).

    Explicitly, in this test we set $f_Y(u_y, x, l; \theta_Y) = \theta_Y[0]e^{-x^2}$,
    which means that $r(x, z, l; theta) = f_Y(\cdot, x, l; \theta_Y)$ since $f_Y$ is a
    constant with respect to $u_y$.

    Note that increasing the value of `n_eval_pts_per_dim` comes with a massive memory
    and time overhead - this dictates the size of the "grid" over which we are
    evaluating the constructed $r$ and its derivative, and this grid is 7D!

    On x32-only machines, the computed partial derivatives for non-$\theta_Y$ variables
    are known to be of approximately 1e-7 size, which means that `jnp.allclose` will
    flag them as "not being close" to 0. This appears to be a numerical-rounding error,
    since enabling x64-precision calculations makes this issue disappear.
    """
    mlps, r_analytic = uy_independent_mlps(k_len=k_len)
    r = ricardo_regression_function(
        k_len=k_len,
        z_len=z_len,
        theta_x=jnp.ones((1,)),
        n_points=n_points,
        **mlps,
    )
    dr_dtheta = jax.grad(r, argnums=1)

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

    r = vectorise_over_dict_args(r, xzl.keys(), theta.keys())
    r_analytic = vectorise_over_dict_args(r_analytic, xzl.keys(), theta.keys())
    dr_dtheta = vectorise_over_dict_args(dr_dtheta, xzl.keys(), theta.keys())

    assert jnp.allclose(r(xzl, theta), r_analytic(xzl, theta))

    dr_evaluation = dr_dtheta(xzl, theta)
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


def test_uy_independent_of_ux(
    ricardo_regression_function,
    ux_independent_mlps,
    vectorise_over_dict_args,
    k_len: int = 5,
    z_len: int = 10,
    n_points: int = 1000,
    n_eval_pts_per_dim: int = 10,
    f_y: MLPAlias = lambda xu_y, theta_y: (xu_y["u_y"] - xu_y["x"]) * theta_y,
) -> None:
    r"""Build a regression function for a system where $U_Y$ is independent of $U_X$.

    This means that, given a functional form for $f_Y$, we can validate the builder is
    working as intended by simply performing the integral that $r$ _should_ be
    evaluating manually, and comparing the results to what we get out of $r$ itself. We
    can even fix the RNG key to ensure that the answers should be _exactly_ the same (to
    within numerical precision, of course).
    """
    mlps, r_direct_integration = ux_independent_mlps(k_len, n_points, f_y)
    r = ricardo_regression_function(
        k_len=k_len,
        z_len=z_len,
        theta_x=0.0,
        **mlps,
        n_points=n_points,
    )

    # Create "grid" across which to evaluate the built functions
    theta = {
        "theta_m": jnp.linspace(-5.0, 5.0, num=n_eval_pts_per_dim),
        "theta_r": jnp.linspace(-5.0, 5.0, num=n_eval_pts_per_dim),
        "theta_pi": jnp.linspace(-5.0, 5.0, num=n_eval_pts_per_dim),
        "theta_y": jnp.linspace(-5.0, 5.0, num=n_eval_pts_per_dim),
    }
    xzl = {
        "x": jnp.linspace(-5.0, 5.0, num=n_eval_pts_per_dim),
        "z": jnp.linspace(-5.0, 5.0, num=n_eval_pts_per_dim),
        "l": jnp.linspace(-5.0, 5.0, num=n_eval_pts_per_dim),
    }

    r = vectorise_over_dict_args(r, xzl.keys(), theta.keys())
    r_direct = vectorise_over_dict_args(r_direct_integration, xzl.keys(), theta.keys())

    assert jnp.allclose(r(xzl, theta), r_direct(xzl, theta))


def test_regression_correctly_calculates_pi_ul(
    ricardo_regression_function,
    k_len: int = 3,
    z_len: int = 1,
    n_points: int = 10,
) -> None:
    r"""The regression function must use one shared vector of mixture probabilities.

    Setting $f_Y=1$ means

    $$
    r(x,z,l)
    =
    \sum_{c=1}^{K}\pi_{ul}(c).
    $$

    Therefore, the result must equal one.
    """
    rng = np.random.default_rng()

    def f_pi(ul: dict, _theta_pi):  # noqa: ARG001
        """Produce a unique array each time"""
        return jnp.asarray(rng.normal(size=k_len))

    def f_ux(xzl: dict, theta_x):
        return xzl["x"] * theta_x

    def f_r(czl: dict, _theta_r):
        return jnp.ones_like(czl["z"])

    def f_m(_czl: dict, _theta_m):
        return jnp.asarray(0.0)

    def f_y(_xuy: dict, _theta_y):
        return jnp.asarray(1.0)

    r = ricardo_regression_function(
        k_len=k_len,
        z_len=z_len,
        f_ux=f_ux,
        f_pi=f_pi,
        f_y=f_y,
        f_r=f_r,
        f_m=f_m,
        theta_x=jnp.ones((1,)),
        n_points=n_points,
    )

    xzl = {
        "x": jnp.asarray(0.5),
        "z": jnp.asarray([0.2]),
        "l": jnp.asarray([0.1]),
    }
    theta = {
        "theta_m": jnp.asarray(0.0),
        "theta_r": jnp.asarray(0.0),
        "theta_pi": jnp.asarray(0.0),
        "theta_y": jnp.asarray(0.0),
    }
    assert jnp.isclose(r(xzl, theta), 1.0)
