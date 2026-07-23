import jax
import jax.numpy as jnp
import pytest

from causalprog.graph.ricardo import (
    ModelParam,
    build_loss_function,
    build_regression_function,
    example_model,
)
from causalprog.quadrature import UniformWeightMonteCarloGaussianQuadrature as UWMCGQuad
from causalprog.solvers.sgd import stochastic_gradient_descent
from causalprog.utils.norms import l2_normsq


@pytest.mark.parametrize("xl", [pytest.param({"x": 1.0, "l": 1.0}, id="x = 1, l = 1")])
def test_integration_reduce_to_linear(
    xl: dict[str, float],
    pytree_allclose,
    rng_key,
    d_z: int = 5,
    k_len: int = 10,
    n_sample_pts: int = 1_000_000,
    x_tilde: float = 1.0,
    delta: float = 0.25,
) -> None:
    # f_m and f_r being set in this way effectively
    # imposes that sigma_czl = 1/2
    def f_m(*args, **kwargs):
        """Note that this results in sigmoid(f_m) = 0.5 always."""
        return 0.0

    def f_r(*args, **kwargs):
        """This results in the 1-vector in R^d_z."""
        return jnp.ones((d_z,))

    # f_pi doesn't really matter since it'll be softmax'd to 1
    def f_pi(*args, **kwargs):
        return jnp.ones((k_len,))

    # Next, setup our 1-point evaluation set
    evaluation_points = {
        "x": jnp.atleast_1d(x_tilde),
        "z": jnp.atleast_1d(0.5),
        "l": jnp.atleast_1d(0.5),
    }
    r_hat_i = jnp.atleast_1d(0.0)
    alpha = 2 * x_tilde**2 + 3.0 / 4.0

    def f_y(u_yxl: dict[str, jax.Array], theta_y: ModelParam) -> jax.Array:
        """"""
        return theta_y / u_yxl["l"] * (u_yxl["u_y"] - u_yxl["x"]) ** 2

    def g(xzl: dict[str, jax.Array], theta_x: ModelParam) -> jax.Array:
        """"""
        e_1 = jnp.zeros((d_z,))
        e_1.at[0].set(1.0)
        return -xzl["x"] * e_1

    # Build model, manual attachment to nodes for now...
    graph = example_model(
        z_len=d_z,
        compute_u_x=g,
        compute_u_y=f_pi,
        compute_x=None,
        compute_phi_x=None,
        compute_y=f_y,
    )
    graph.get_node("u_y").f_r = f_r
    graph.get_node("u_y").f_m = f_m

    # Construct the regression function
    regression_function = build_regression_function(
        graph,
        theta_x=jnp.atleast_1d(0.0),
        quadrature=UWMCGQuad(n_points=n_sample_pts, rng_key=rng_key),
    )

    def r_analytic(xzl: dict[str, jax.Array], theta: ModelParam) -> jax.Array:
        """Analytic form of the regression function we expect."""
        return theta["theta_y"] * alpha / xzl["l"]

    # Learn the set of optimal parameters, which should be theta_y = 0 at
    # B(\theta) = 0.
    theta_0 = {
        "theta_y": 0.01,
        "theta_pi": 0.0,
        "theta_m": 0.0,
        "theta_r": 0.0,
        "theta_x": 0.0,
    }
    expected_theta_star = dict(theta_0)
    expected_theta_star["theta_y"] = 0.0
    loss_function = build_loss_function(regression_function, evaluation_points, r_hat_i)

    learn_initialiser_result = stochastic_gradient_descent(loss_function, theta_0)

    assert pytree_allclose(learn_initialiser_result.fn_args, expected_theta_star)
    assert jnp.allclose(learn_initialiser_result.obj_val, 0.0)

    # This is where I'd construct d, if I could.
    # But we do have the analytic form at least...
    def d_analytic(xl: dict[str, jax.Array], theta: ModelParam) -> jax.Array:
        """Analytic form of the causal response that we expect."""
        return theta["theta_y"] / xl["l"] * (1 + xl["x"] ** 2)

    # And this is where I do my optimisation now.
    epsilon = delta**2
    b_theta_star = learn_initialiser_result.obj_val

    expected_theta_y_max = delta * xl["l"] / alpha
    expected_theta_y_min = -expected_theta_y_max

    expected_d_max = delta * (1 + xl["x"] ** 2) / alpha
    expected_d_min = -expected_d_max

    # Loss function should be = epsilon at the expected ends of the interval... right?

    # Contraint is wrong, it's only a penalty when we are above the threshold...
    def lagrangian(theta_lmult):
        theta, lmult = theta_lmult
        return d_analytic(xl, theta) - lmult * jnp.abs(loss_function(theta) - epsilon)

    grad_lagrangian = jax.grad(lagrangian)

    def objective(theta_lmult):
        return l2_normsq(grad_lagrangian(theta_lmult))

    near_minimum_theta_guess = dict(theta_0)
    near_minimum_theta_guess["theta_y"] = expected_theta_y_min
    result = stochastic_gradient_descent(objective, (near_minimum_theta_guess, 1.0))
