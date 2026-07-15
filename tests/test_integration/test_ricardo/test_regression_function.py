from collections.abc import Iterable

import jax
import jax.numpy as jnp

from causalprog.graph.ricardo import MLPAlias
from causalprog.quadrature import UniformWeightMonteCarloGaussianQuadrature as UWMCGQuad


def _vectorise_over_dict_args(f: MLPAlias, *dict_keys: Iterable[str]) -> MLPAlias:
    """Vectorise a pure function of dictionary arguments across the dictionary keys.

    This is essentially a wrapper around iterative applications of `jax.vmap` with the
    appropriate `in_axes` specified. The net effect is that if the input `f` was
    being called with a dictionary argument, whose keys were scalar-valued, the returned
    function can be called with the same dictionary argument whose keys are
    vector-valued, and returns a vector-valued output.

    Note that all vmap-ing is done along axis 0. If you want to pass in vector-values
    for some of the dictionary key inputs, ensure that they are aligned along the
    correct axis (each _row_ should be one value of the input).
    """
    vec_f = f
    all_keys = [key for key_list in dict_keys for key in key_list]
    for key in all_keys:
        vec_f = jax.vmap(
            vec_f,
            in_axes=tuple(
                {k: None if k != key else 0 for k in arg_keys} for arg_keys in dict_keys
            ),
        )
    return vec_f


def test_fy_independent_of_uy(
    jax_enable_x64,  # noqa: ARG001
    ricardo_regression_function,
    k_len: int = 5,
    z_len: int = 10,
    n_points: int = 1000,
    n_eval_pts_per_dim: int = 10,
    f_r: MLPAlias = lambda czl, theta_m: theta_m * jnp.ones_like(czl["z"]),
    f_m: MLPAlias = lambda czl, theta_r: czl["c"] * theta_r,
    f_ux: MLPAlias = lambda xzl, theta_x: xzl["x"] * theta_x[0],
    f_y: MLPAlias = lambda xu_y, theta_y: theta_y[0] * jnp.exp(-(xu_y["x"] ** 2)),
    r_analytic: MLPAlias = lambda xzl, theta: (
        theta["theta_y"][0] * jnp.exp(-(xzl["x"] ** 2))
    ),
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
    r = ricardo_regression_function(
        k_len=k_len,
        z_len=z_len,
        f_ux=f_ux,
        f_pi=lambda ucl, theta_pi: theta_pi * ucl["c"] + ucl["u_x"] * jnp.arange(k_len),
        f_y=f_y,
        f_r=f_r,
        f_m=f_m,
        theta_x=jnp.ones((1,)),
        n_points=n_points,
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

    r = _vectorise_over_dict_args(r, xzl.keys(), theta.keys())
    r_analytic = _vectorise_over_dict_args(r_analytic, xzl.keys(), theta.keys())
    dr_dtheta = _vectorise_over_dict_args(dr_dtheta, xzl.keys(), theta.keys())

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
    rng_key,
    ricardo_regression_function,
    k_len: int = 5,
    z_len: int = 10,
    n_points: int = 1000,
    n_eval_pts_per_dim: int = 10,
    f_r: MLPAlias = lambda czl, _: jnp.ones_like(czl["z"]),
    f_m: MLPAlias = lambda _, __: -float("inf"),
    f_ux: MLPAlias = lambda xzl, theta_x: xzl["x"] * theta_x,
    f_y: MLPAlias = lambda xu_y, theta_y: (xu_y["u_y"] - xu_y["x"]) * theta_y,
) -> None:
    r"""Build a regression function for a system where $U_Y$ is independent of $U_X$.

    In this example, we set
    - $f_{\pi} = 1 \theta_{\pi}$ where $1\in\mathbb{R}^K,
    - $f_r = 1\in\mathbb{R}^{d_z}$,
    - $f_m = -\infty$

    which has the effect of ensuring that $s_q v_y + m_y = s_q$, meaning that the
    regression function

    $$
    r(x, z, l) = \int f_Y(u_y, x; \theta_Y)p_N(u_y; 0, 1) \mathrm{d}u_y
    = \mathbb{E}[f_Y(U, x; \theta_Y)]
    $$

    where the expectation is with respect to $U\sim\mathcal{N}(0,1)$.

    This means that, given a functional form for $f_Y$, we can validate the builder is
    working as intended by simply performing the integral that $r$ _should_ be
    evaluating manually, and comparing the results to what we get out of $r$ itself. We
    can even fix the RNG key to ensure that the answers should be _exactly_ the same (to
    within numerical precision, of course).
    """
    r = ricardo_regression_function(
        k_len=k_len,
        z_len=z_len,
        f_ux=f_ux,
        f_pi=lambda _, theta_pi: jnp.full((k_len,), theta_pi),
        f_y=f_y,
        f_r=f_r,
        f_m=f_m,
        theta_x=0.0,
        n_points=n_points,
    )

    # What r _should_ be doing is just integrating f_Y with the appropriate scheme.
    # So let's do this here, and compare results.
    def r_direct_integration(xzl: dict, theta: dict):
        return UWMCGQuad(n_points, rng_key=rng_key).integrate(
            lambda u_y: f_y({"u_y": u_y, **xzl}, theta["theta_y"]),
            -float("inf"),
            float("inf"),
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

    r = _vectorise_over_dict_args(r, xzl.keys(), theta.keys())
    r_direct = _vectorise_over_dict_args(r_direct_integration, xzl.keys(), theta.keys())

    assert jnp.allclose(r(xzl, theta), r_direct(xzl, theta))
