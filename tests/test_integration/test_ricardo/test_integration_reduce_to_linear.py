import jax
import jax.numpy as jnp

from causalprog.graph import Graph
from causalprog.graph.ricardo import (
    MLPAlias,
    ModelParam,
    build_regression_function,
    example_model,
)
from causalprog.quadrature import UniformWeightMonteCarloGaussianQuadrature as UWMCGQuad


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
    - $g(x, z, l) = -x e_1$ where $e_1$ is the axis-1 unit vector.
    - $f_y$ is defined by `f_y`, above.
    """

    def f_m(*args, **kwargs):
        """Note that this results in sigmoid(f_m) = 0.5 always."""
        return 0.0

    def f_r(*args, **kwargs):
        """This results in the 1-vector in R^d_z."""
        return jnp.ones((d_z,))

    def f_pi(*args, **kwargs):
        return jnp.ones((k_len,))

    def g(xzl: dict[str, jax.Array], _: ModelParam) -> jax.Array:
        """"""
        e_1 = jnp.zeros((d_z,))
        e_1.at[0].set(1.0)
        return -xzl["x"] * e_1

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


# TODO Monday: figure out whether it's sqrt(v_y) or not. I think it is. Almost surely even. But use something easy to check???


def test_integration_reduce_to_linear(
    jax_enable_x64,
    rng_key,
    d_z: int = 5,
    k_len: int = 10,
    n_sample_pts: int = 1_000_000,
) -> None:
    graph = graph_for_example(d_z, k_len)

    # Construct the regression function
    regression_function = build_regression_function(
        graph,
        theta_x=jnp.atleast_1d(0.0),
        quadrature=UWMCGQuad(n_points=n_sample_pts, rng_key=rng_key),
        # lower_domain_limit=-1000.0,
        # upper_domain_limit=1000.0,
    )

    xzl_to_plot_at = {
        "x": 0.0,
        "z": 0.0,  # confirmed to not matter
        "l": 0.5,
    }
    theta_range = {
        "theta_y": jnp.linspace(-5, 5, num=100),
        "theta_pi": 0.0,
        "theta_m": 0.0,
        "theta_r": 0.0,
        "theta_x": 0.0,
    }
    analytic_r_to_plot = jax.vmap(
        lambda theta: r_analytic(xzl_to_plot_at, theta),
        in_axes=(
            {
                "theta_y": 0,
                "theta_pi": None,
                "theta_m": None,
                "theta_r": None,
                "theta_x": None,
            },
        ),
    )(theta_range)
    built_r_to_plot = jax.vmap(
        lambda theta: regression_function(xzl_to_plot_at, theta),
        in_axes=(
            {
                "theta_y": 0,
                "theta_pi": None,
                "theta_m": None,
                "theta_r": None,
                "theta_x": None,
            },
        ),
    )(theta_range)

    analytic_over_built = analytic_r_to_plot / built_r_to_plot

    print(
        xzl_to_plot_at,
        "Mean:",
        analytic_over_built.mean(),
        "Std:",
        analytic_over_built.std(),
    )
