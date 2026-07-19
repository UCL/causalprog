from collections.abc import Callable
from itertools import product

import jax
import jax.numpy as jnp

from causalprog.graph import Graph
from causalprog.graph.ricardo import (
    MLPAlias,
    build_causal_response_function,
    example_model,
)
from causalprog.quadrature import (
    UniformWeightMonteCarloGaussianQuadrature as UWMCGQuad,
)

from ._helpers import _vectorise_over_dict_args


def _constant_zero(*_args: object, **_kwargs: object) -> float:
    """Return zero for graph functions that are irrelevant to d."""
    return 0.0


def _build_test_graph(f_y: MLPAlias) -> Graph:
    """Build Ricardo's graph with only the outcome function being relevant."""
    return example_model(
        compute_u_x=_constant_zero,
        compute_u_y=_constant_zero,
        compute_phi_x=_constant_zero,
        compute_x=_constant_zero,
        compute_y=f_y,
    )


def _build_test_causal_response_function(
    f_y: MLPAlias,
    n_points: int,
    rng_key: jax.Array,
) -> Callable:
    """Build a causal response function for testing."""
    graph = _build_test_graph(f_y)
    quadrature = UWMCGQuad(n_points, rng_key=rng_key)

    return build_causal_response_function(
        graph,
        quadrature,
    )


def test_fy_independent_of_uy(
    rng_key: jax.Array,
    n_points: int = 100,
    n_eval_pts_per_dim: int = 10,
) -> None:
    r"""If $f_Y$ is independent of $U_Y$, d should equal $f_Y$ exactly."""

    def f_y(u_yxl: dict, theta_y: dict) -> jax.Array:
        return theta_y["x_scale"] * jnp.exp(-(u_yxl["x"] ** 2)) + theta_y[
            "l_scale"
        ] * jnp.sum(u_yxl["l"])

    d = _build_test_causal_response_function(
        f_y,
        n_points,
        rng_key,
    )

    def d_from_theta_y(
        xl: dict,
        theta_y: dict,
    ) -> jax.Array:
        return d(xl, {"theta_y": theta_y})

    def d_analytic(
        xl: dict,
        theta_y: dict,
    ) -> jax.Array:
        return theta_y["x_scale"] * jnp.exp(-(xl["x"] ** 2)) + theta_y[
            "l_scale"
        ] * jnp.sum(xl["l"])

    eval_values = jnp.linspace(
        -1.0,
        1.0,
        num=n_eval_pts_per_dim,
    )

    xl = {
        "x": eval_values,
        "l": jnp.stack(
            [
                eval_values,
                jnp.roll(eval_values, shift=1),
            ],
            axis=1,
        ),
    }
    theta_y = {
        "x_scale": eval_values,
        "l_scale": eval_values,
    }

    d_grid = _vectorise_over_dict_args(
        d_from_theta_y,
        xl.keys(),
        theta_y.keys(),
    )
    analytic_grid = _vectorise_over_dict_args(
        d_analytic,
        xl.keys(),
        theta_y.keys(),
    )

    actual = d_grid(xl, theta_y)
    expected = analytic_grid(xl, theta_y)

    assert jnp.allclose(actual, expected)


def test_causal_response_matches_standard_normal_moments(
    rng_key: jax.Array,
    jax_enable_x64,  # noqa: ARG001
    n_points: int = 1_000_000,
    n_eval_pts_per_dim: int = 4,
    eval_range: tuple[float, float] = (-10.0, 10.0),
    grid_batch_size: int = 64,
    atol: float = 0.0,
    rtol: float = 0.01,
) -> None:
    r"""Compare d against known moments of a standard-normal latent variable.

    For U_Y ~ N(0, 1),

    E[U_Y] = 0
    and
    E[U_Y**2] = 1.

    The Cartesian product of input values is evaluated in batches to limit
    peak memory usage.
    """

    def f_y(u_yxl: dict, theta_y: dict) -> jax.Array:
        return (
            theta_y["quadratic"] * u_yxl["u_y"] ** 2
            + theta_y["linear"] * u_yxl["u_y"]
            + theta_y["x_weight"] * u_yxl["x"]
            + theta_y["l_weight"] * jnp.sum(u_yxl["l"])
        )

    d = _build_test_causal_response_function(
        f_y,
        n_points,
        rng_key,
    )

    eval_values = jnp.linspace(
        *eval_range,
        num=n_eval_pts_per_dim,
    )
    l_values = jnp.stack(
        [
            eval_values,
            jnp.roll(eval_values, shift=1),
        ],
        axis=1,
    )

    # Each row selects:
    # x, l, quadratic, linear, x_weight, l_weight.
    grid_indices = jnp.asarray(
        list(
            product(
                range(n_eval_pts_per_dim),
                repeat=6,
            )
        ),
        dtype=jnp.int32,
    )

    def evaluate_one(
        indices: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        (
            x_index,
            l_index,
            quadratic_index,
            linear_index,
            x_weight_index,
            l_weight_index,
        ) = indices

        xl = {
            "x": eval_values[x_index],
            "l": l_values[l_index],
        }
        theta_y = {
            "quadratic": eval_values[quadratic_index],
            "linear": eval_values[linear_index],
            "x_weight": eval_values[x_weight_index],
            "l_weight": eval_values[l_weight_index],
        }

        actual = d(
            xl,
            {"theta_y": theta_y},
        )
        expected = (
            theta_y["quadratic"]
            + theta_y["x_weight"] * xl["x"]
            + theta_y["l_weight"] * jnp.sum(xl["l"])
        )

        return actual, expected

    evaluate_batch = jax.jit(jax.vmap(evaluate_one))

    for batch_start in range(
        0,
        len(grid_indices),
        grid_batch_size,
    ):
        batch_indices = grid_indices[batch_start : batch_start + grid_batch_size]

        actual, expected = evaluate_batch(batch_indices)

        assert bool(
            jnp.allclose(
                actual,
                expected,
                atol=atol,
                rtol=rtol,
            )
        )


def test_causal_response_is_jittable_and_differentiable(
    rng_key: jax.Array,
    n_points: int = 100,
) -> None:
    """The constructed function should support JIT and differentiation."""

    def f_y(u_yxl: dict, theta_y: dict) -> jax.Array:
        return (theta_y["scale"] ** 2) * (u_yxl["x"] + 2.0 * jnp.sum(u_yxl["l"]))

    d = _build_test_causal_response_function(
        f_y,
        n_points,
        rng_key,
    )
    d_jit = jax.jit(d)

    xl = {
        "x": jnp.asarray(1.5),
        "l": jnp.asarray([2.0, -0.5]),
    }
    model_params = {
        "theta_y": {
            "scale": jnp.asarray(3.0),
        },
        "theta_m": {
            "unused": jnp.asarray(4.0),
        },
    }

    scale = model_params["theta_y"]["scale"]
    input_term = xl["x"] + 2.0 * jnp.sum(xl["l"])

    expected_value = scale**2 * input_term
    expected_derivative = 2.0 * scale * input_term

    value, gradients = jax.value_and_grad(
        d_jit,
        argnums=1,
    )(xl, model_params)

    assert jnp.isclose(value, expected_value)
    assert jnp.isclose(
        gradients["theta_y"]["scale"],
        expected_derivative,
    )


# TODO: some sort of batching of the integral to prevent inf?
