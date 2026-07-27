import jax
import jax.numpy as jnp

from causalprog.graph import Graph
from causalprog.graph.ricardo import (
    MLPAlias,
    ModelParam,
    build_loss_function,
    build_regression_function,
    example_model,
)
from causalprog.quadrature import UniformWeightMonteCarloGaussianQuadrature as UWMCGQuad
from causalprog.solvers.sgd import stochastic_gradient_descent


def alpha(x: float):
    r"""The value of $\alpha(x) = \frac{3}{4}(1 + 3x^2)$."""
    return (3.0 / 4.0) * (1.0 + 3.0 * x**2)


def f_y(u_yxl: dict[str, jax.Array], theta_y: ModelParam) -> jax.Array:
    r"""$f_Y(u_y, x, l; \theta_Y) = \frac{\theta_Y}{l}(u_y - x)^2$."""
    return (theta_y / u_yxl["l"]) * (u_yxl["u_y"] - u_yxl["x"]) ** 2


def r_analytic(xzl: dict[str, jax.Array], theta: ModelParam) -> jax.Array:
    r"""Expected analytic form of the regression function;

    $$ r(x, z, l; \theta) = \frac{\theta_Y\alpha(x)}{l}. $$
    """
    return theta["theta_y"] * alpha(xzl["x"]) / xzl["l"]


def mlps_for_example(d_z: int, k_len: int) -> dict[str, MLPAlias]:
    r"""Constructs MLP-stand-ins used in this problem, returning them in
    a dictionary key-ed by name.#

    - $f_m$ returns 0 so that the sigmoid it's passed into always returns 0.5.
    - $f_r$ just returns a 1-vector of appropriate length.
    - $f_{pi}$ also just returns a 1-vector.
    - $g(x, z, l) = -x \mathbb{I}$ where $\mathbb{I}$ is the $\mathbb{R}^{d_z}$ unit
        vector with identical elements.
    - $f_y$ is defined by `f_y`, above.
    """

    def f_m(*args, **kwargs):
        """Note that this results in sigmoid(f_m) = 0.5 always."""
        return 0.0

    def f_r(*args, **kwargs):
        """This results in the 1-vector in R^d_z after passing through tanh."""
        return jnp.full((d_z,), float("inf"))

    def f_pi(*args, **kwargs):
        """Theoretically irrelevant as it will be softmax'd."""
        return jnp.ones((k_len,))

    def g(xzl: dict[str, jax.Array], _: ModelParam) -> jax.Array:
        """This form ensures that m_y^T g gives us a mean of -x/2."""
        return -xzl["x"] * jnp.ones((d_z,)) / jnp.sqrt(d_z)

    return {"f_r": f_r, "f_m": f_m, "f_pi": f_pi, "g": g, "f_y": f_y}


def graph_for_example(d_z: int, k_len: int) -> Graph:
    """Construct Ricardo's graph, as it would appear for this example."""
    mlps = mlps_for_example(d_z=d_z, k_len=k_len)

    # Build model, manual attachment to nodes for now...
    graph = example_model(
        z_len=d_z,
        compute_u_x=mlps["g"],
        compute_u_y=mlps["f_pi"],
        compute_x=None,
        compute_phi_x=None,
        compute_y=mlps["f_y"],
    )
    graph.get_node("u_y").f_r = mlps["f_r"]
    graph.get_node("u_y").f_m = mlps["f_m"]

    return graph


def d_analytic(xl: dict[str, jax.Array], theta: ModelParam) -> jax.Array:
    r"""$d(x, l; \theta) = \frac{\theta_Y}{l}(1 + x^2)$."""
    return theta["theta_y"] / xl["l"] * (1.0 + xl["x"] ** 2)


def test_integration_reduce_to_linear(
    jax_enable_x64,  # noqa: ARG001
    pytree_allclose,
    rng_key,
    d_z: int = 5,
    k_len: int = 10,
    n_sample_pts: int = 1_000_000,
) -> None:
    """This regression test follows the example in
    `docs/theory/reduce-to-linear-example.md`.
    """
    graph = graph_for_example(d_z, k_len)

    # Construct the regression function
    regression_function = build_regression_function(
        graph,
        theta_x=jnp.atleast_1d(0.0),
        quadrature=UWMCGQuad(n_points=n_sample_pts, rng_key=rng_key),
        domain_lower_bound=-1000.0,
        domain_upper_bound=1000.0,
    )

    x_tilde = 0.0
    evaluation_points = {
        "x": jnp.atleast_1d(x_tilde),
        "z": jnp.atleast_1d(0.0),
        "l": jnp.atleast_1d(0.5),
    }
    r_hat_i = jnp.atleast_1d(0.0)
    # Determine the learnt initialiser, theta_star.
    # This should be theta_star = {theta_y: 0.0},
    # other theta values are irrelevant.
    loss_function = build_loss_function(regression_function, evaluation_points, r_hat_i)
    theta_y_opt = 0.0

    # We should now be able to "optimise" the loss function to find the learnt
    # initialiser for theta...
    independent_param_starting_values = {
        "theta_pi": 0.0,
        "theta_r": 0.0,
        "theta_m": 0.0,
    }
    theta_y_initial_guess = 1.0
    optimisation_result = stochastic_gradient_descent(
        loss_function,
        {"theta_y": theta_y_initial_guess, **independent_param_starting_values},
    )
    theta_star = optimisation_result.fn_args

    assert optimisation_result.successful
    assert pytree_allclose(
        theta_star,
        {"theta_y": theta_y_opt, **independent_param_starting_values},
    )
    assert jnp.allclose(0.0, optimisation_result.obj_val)

    # And now we should be solving a simple optimisation problem...
    delta = 0.5
    epsilon = delta**2
