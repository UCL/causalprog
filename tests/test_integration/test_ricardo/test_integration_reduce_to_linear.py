import jax
import jax.numpy as jnp

from causalprog.graph import Graph
from causalprog.graph.ricardo import (
    MLPAlias,
    ModelParam,
    build_causal_response_function,
    build_loss_function,
    build_regression_function,
    example_model,
)
from causalprog.quadrature import UniformWeightMonteCarloGaussianQuadrature as UWMCGQuad
from causalprog.solvers.augmented_lagrangian import maximise, minimise
from causalprog.solvers.sgd import stochastic_gradient_descent


def alpha(x: float):
    r"""The value of $\alpha(x) = \frac{3}{4}(1 + 3x^2)$."""
    return (3.0 / 4.0) * (1.0 + 3.0 * x**2)


def r_analytic(xzl: dict[str, jax.Array], theta: ModelParam) -> jax.Array:
    r"""Expected analytic form of the regression function;

    $$ r(x, z, l; \theta) = \frac{\theta_Y\alpha(x)}{l}. $$
    """
    return theta["theta_y"] * alpha(xzl["x"]) / xzl["l"]


def loss_analytic(evaluation_pt: dict[str, jax.Array], theta: ModelParam) -> jax.Array:
    r"""Analytic form of the loss function,
    $B(\theta) = \frac{\theta_Y^2 \alpha(\tilde{x})^2}{\tilde{l}^2}.$
    """
    return r_analytic(evaluation_pt, theta) ** 2


def d_analytic(xl: dict[str, jax.Array], theta: ModelParam) -> jax.Array:
    r"""$d(x, l; \theta) = \frac{\theta_Y}{l}(1 + x^2)$."""
    return theta["theta_y"] * (1.0 + xl["x"] ** 2) / xl["l"]


def analytic_solution(
    xl: jax.Array, evaluation_pt: jax.Array, delta: float
) -> dict[str, jax.Array]:
    """Optimal solution values for the problem."""
    alpha_x_tilde = alpha(evaluation_pt["x"])
    argmin = -delta * evaluation_pt["l"] / alpha_x_tilde
    min_val = -(delta * evaluation_pt["l"] / xl["l"]) * (
        (1 + xl["x"] ** 2) / alpha_x_tilde
    )
    l_mult = (
        (evaluation_pt["l"] / xl["l"])
        * (1 + xl["x"] ** 2)
        / (2 * delta * alpha_x_tilde)
    )

    return {
        "argmin": argmin,
        "argmax": -argmin,
        "min_val": min_val,
        "max_val": -min_val,
        "l_mult": l_mult,
    }


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

    def f_y(u_yxl: dict[str, jax.Array], theta_y: ModelParam) -> jax.Array:
        r"""$f_Y(u_y, x, l; \theta_Y) = \frac{\theta_Y}{l}(u_y - x)^2$."""
        return (theta_y / u_yxl["l"]) * (u_yxl["u_y"] - u_yxl["x"]) ** 2

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


def test_integration_reduce_to_linear(
    jax_enable_x64,  # noqa: ARG001
    pytree_allclose,
    rng_key,
    d_z: int = 5,
    k_len: int = 10,
    n_sample_pts: int = 1_000_000,
    learn_initialiser_theta_y_guess: float = 0.5,
    delta: float = 0.5,
    independent_params: tuple[str, ...] = ("theta_pi", "theta_r", "theta_m"),
) -> None:
    """This regression test follows the example in
    `docs/theory/reduce-to-linear-example.md`.
    """
    quad_method = UWMCGQuad(n_points=n_sample_pts, rng_key=rng_key)
    indep_params = dict.fromkeys(independent_params, 0.0)
    x_tilde = 0.1
    evaluation_points = {
        "x": jnp.atleast_1d(x_tilde),
        "z": jnp.atleast_1d(0.0),
        "l": jnp.atleast_1d(1.0),
    }
    r_hat_i = jnp.atleast_1d(0.0)
    theta_y_opt = 0.0
    xl_to_solve_at = {"x": 2.0, "l": 2.0}
    expected_solution = analytic_solution(xl_to_solve_at, evaluation_points, delta)

    graph = graph_for_example(d_z, k_len)

    # Construct the regression function
    regression_function = build_regression_function(
        graph,
        theta_x=jnp.atleast_1d(0.0),
        quadrature=quad_method,
        domain_lower_bound=-100.0,
        domain_upper_bound=100.0,
    )

    # Determine the learnt initialiser, theta_star.
    # This should be theta_star = {theta_y: 0.0},
    # other theta values are irrelevant.
    loss_function = build_loss_function(regression_function, evaluation_points, r_hat_i)

    # We should now be able to "optimise" the loss function to find the learnt
    # initialiser for theta...
    learn_initaliser = stochastic_gradient_descent(
        loss_function,
        {"theta_y": learn_initialiser_theta_y_guess, **indep_params},
    )
    theta_star = learn_initaliser.fn_args

    assert learn_initaliser.successful
    assert pytree_allclose(
        theta_star,
        {"theta_y": theta_y_opt, **indep_params},
    )
    assert jnp.allclose(0.0, learn_initaliser.obj_val)

    # And now we should be solving a simple optimisation problem...
    # I guess just directly attack the Lagrangian?
    epsilon = delta**2
    response_function = build_causal_response_function(graph, quad_method)

    min_result = minimise(
        response_function,
        lambda _, theta: loss_function(theta) - learn_initaliser.obj_val,
        bounds_epsilon=epsilon,
        initial_guess={
            "theta_y": expected_solution["argmin"][0],
            **indep_params,
        },
        parameter_values=xl_to_solve_at,
        n_iter=6,
        update_mu=lambda mu: 3.0 * mu,
        update_learning_rate=lambda lr, mu, _: lr / jnp.sqrt(mu),
    )
    max_result = maximise(
        response_function,
        lambda _, theta: loss_function(theta) - learn_initaliser.obj_val,
        bounds_epsilon=epsilon,
        initial_guess={
            "theta_y": expected_solution["argmax"][0],
            **indep_params,
        },
        parameter_values=xl_to_solve_at,
        n_iter=6,
        update_mu=lambda mu: 3.0 * mu,
        update_learning_rate=lambda lr, mu, _: lr / jnp.sqrt(mu),
    )

    assert min_result.successful
    assert max_result.successful
    # Replace me with actual checks, but for now we're printing just to see how well
    # we do
    print()
    print(
        "\nMin |",
        min_result.fn_args["theta_y"],
        "expected:",
        expected_solution["argmin"][0],
    )
    print(
        "Max |",
        max_result.fn_args["theta_y"],
        "expected:",
        expected_solution["argmax"][0],
    )
