import jax
import jax.numpy as jnp
import pytest

from causalprog.graph.ricardo import MLPAlias, ModelParam, build_loss_function
from causalprog.solvers.sgd import stochastic_gradient_descent


@pytest.mark.parametrize(
    ("call_initialiser_at", "expected_residual"),
    [
        pytest.param({"a": 1.0, "b": 0.0, "c": -1.0}, 0.0, id="Call at optimal theta"),
        pytest.param(
            {"a": 0.0, "b": 1.0, "c": -1.0}, 0.0, id="Call at equivalent optimal theta"
        ),
        pytest.param(
            {"a": 1.0, "b": 0.0, "c": -0.5},
            74.0 / 64.0 / 5.0,
            id="Actually evaluates residual",
        ),
    ],
)
def test_learn_initialiser_deterministic_fn(
    call_initialiser_at: ModelParam,
    expected_residual: float,
    r: MLPAlias = lambda data, theta: (
        (data["x"] - theta["a"]) * (data["x"] - theta["b"]) * (data["x"] - theta["c"])
    ),
) -> None:
    """Simple explicit-evaluation test for the function constructed by
    `learn_initialiser`.
    """
    theta_opt = {"a": 1.0, "b": 0.0, "c": -1.0}
    eval_pts = {"x": jnp.linspace(-1.0, 1.0, num=5)}
    r_hat_pts = r(eval_pts, theta_opt)

    learn_initialiser = build_loss_function(r, eval_pts, r_hat_pts)
    residual = learn_initialiser(call_initialiser_at)

    assert jnp.allclose(residual, expected_residual)


@pytest.mark.parametrize(
    ("axes_mapping", "expected_values"),
    [
        pytest.param(
            None,
            jnp.array(
                (11.0 / 100.0) ** 2
                * ((0 + 1 + 2) ** 2 + (3 + 4 + 5) ** 2 + (6 + 7 + 8) ** 2)
                / 3
            ),
            id="Default map along axes 0",
        ),
        pytest.param(
            {"x": 0, "y": 0},
            jnp.array(
                (11.0 / 100.0) ** 2
                * ((0 + 1 + 2) ** 2 + (3 + 4 + 5) ** 2 + (6 + 7 + 8) ** 2)
                / 3
            ),
            id="Explicit map along axes 0",
        ),
        pytest.param(
            {"x": 1, "y": 1},
            jnp.array(
                (11.0 / 100.0) ** 2
                * ((0 + 3 + 6) ** 2 + (1 + 4 + 7) ** 2 + (2 + 5 + 8) ** 2)
                / 3
            ),
            id="Map along axes 1",
        ),
        pytest.param(
            {"x": 1},
            jnp.array(
                (
                    ((0 + 3 + 6) / 10 + (0 + 1 + 2) / 100) ** 2
                    + ((1 + 4 + 7) / 10 + (3 + 4 + 5) / 100) ** 2
                    + ((2 + 5 + 8) / 10 + (6 + 7 + 8) / 100) ** 2
                )
                / 3
            ),
            id="x along 1, y along 0",
        ),
        pytest.param(
            {"y": 1},
            jnp.array(
                (
                    ((0 + 3 + 6) / 100 + (0 + 1 + 2) / 10) ** 2
                    + ((1 + 4 + 7) / 100 + (3 + 4 + 5) / 10) ** 2
                    + ((2 + 5 + 8) / 100 + (6 + 7 + 8) / 10) ** 2
                )
                / 3
            ),
            id="x along 0, y along 1",
        ),
    ],
)
def test_learn_initialiser_evaluation_points_axes_mapping(
    axes_mapping: dict[str, int],
    expected_values: jax.Array,
    r: MLPAlias = lambda data, _: data["x"].sum() + data["y"].sum(),
) -> None:
    """Check that the axes mapping for input evaluation points is respected.

    To do so, we use a fixed r-function and let the r_hat_i points all be 0.
    This effectively gives us a function B that is just the sum of the squares
    of the r-function at the evaluation points times a constant, which we can then
    check the value of (when evaluated) to confirm that the evaluation points are mapped
    correctly.
    """
    eval_pts = {
        "x": 0.1 * jnp.arange(9).reshape(3, 3),
        "y": 0.01 * jnp.arange(9).reshape(3, 3),
    }
    if axes_mapping is not None:
        n_eval_pts = eval_pts["x"].shape[axes_mapping.get("x", 0)]
    else:
        n_eval_pts = eval_pts["x"].shape[0]

    r_hat_pts = jnp.zeros((n_eval_pts,))

    learn_initialiser = build_loss_function(
        r,
        eval_pts,
        r_hat_pts,
        evaluation_points_axes_mapping=axes_mapping,
    )

    assert jnp.allclose(learn_initialiser({}), expected_values)


@pytest.mark.parametrize(
    (
        "initial_guess",
        "opt_kwargs",
        "independent_param_atol",
    ),
    [
        pytest.param(
            {
                "theta_r": 1.0,
                "theta_m": 1.0,
                "theta_pi": 1.0,
                "theta_y": jnp.atleast_1d(2.0),
            },
            None,
            1e-16,
            id="Starting at the solution immediately exits with precise results",
        ),
        pytest.param(
            {
                "theta_r": 1.0,
                "theta_m": 1.0,
                "theta_pi": 1.0,
                "theta_y": jnp.atleast_1d(2.1),
            },
            None,
            1e-16,
            id="Independent params should be invariant if starting close to solution",
        ),
        pytest.param(
            {
                "theta_r": 1.0,
                "theta_m": 1.0,
                "theta_pi": 0.0,
                "theta_y": jnp.atleast_1d(20.0),
            },
            {"learning_rate": 1.0},
            1e-4,
            id="Independent params vary for big learning rates / poor initial guess",
        ),
    ],
)
def test_learn_initialiser_uy_independent_regression_fn(
    jax_enable_x64,  # noqa: ARG001
    ricardo_regression_function,
    uy_independent_mlps,
    pytree_allclose,
    pytree_all_same_shape,
    initial_guess: ModelParam,
    opt_kwargs: dict | None,
    independent_param_atol: float,
    theta_y_solution_value: float = 2.0,
    k_len: int = 5,
    z_len: int = 10,
    n_points: int = 1000,
    n_eval_pts: int = 50,
) -> None:
    r"""Test that `learn_optimiser` correctly minimises the function $B$ when provided a
    regression function $r$, formed from an $f_Y$ that is independent of $U_Y$.

    In this test, we have that $r(x, z, l; \theta) = \theta_Y[0]e^{-x^2}$.
    """
    if opt_kwargs is None:
        opt_kwargs = {}
    theta_y_solution = jnp.atleast_1d(theta_y_solution_value)

    mlps, r_analytic = uy_independent_mlps(k_len=k_len)
    r = ricardo_regression_function(
        k_len=k_len,
        z_len=z_len,
        theta_x=jnp.ones((1,)),
        n_points=n_points,
        **mlps,
    )

    evaluation_points = {
        "x": jnp.linspace(-1.0, 1.0, num=n_eval_pts),
        "z": jnp.arange(n_eval_pts),
        "l": 0.0,
    }
    evaluation_points_mapping = {"x": 0, "z": 0, "l": None}
    r_hat_pts = jax.vmap(r_analytic, in_axes=(evaluation_points_mapping, None))(
        evaluation_points, {"theta_y": theta_y_solution}
    )

    expected_solution = dict(initial_guess)
    expected_solution["theta_y"] = theta_y_solution
    learn_initialiser = build_loss_function(
        r,
        evaluation_points,
        r_hat_pts,
        evaluation_points_axes_mapping=evaluation_points_mapping,
    )
    result = stochastic_gradient_descent(
        learn_initialiser,
        initial_guess,
        **opt_kwargs,
    )

    assert result.successful
    assert pytree_all_same_shape(result.fn_args, expected_solution)
    assert jnp.allclose(result.fn_args["theta_y"], theta_y_solution)
    # Other parameters may have moved around slightly due to autodiff imprecisions,
    # though in reality they should not have moved. Be gracious if necessary.
    assert pytree_allclose(
        {k: v for k, v in result.fn_args.items() if k != "theta_y"},
        {k: v for k, v in expected_solution.items() if k != "theta_y"},
        atol=independent_param_atol,
    )


@pytest.mark.parametrize(
    (
        "f_y",
        "initial_theta_y_guess",
        "expected_theta_y",
        "opt_kwargs",
        "independent_param_atol",
    ),
    [
        pytest.param(
            lambda _, theta_y: theta_y,
            1.0,
            1.1,
            {},
            1e-12,
            id="f_Y returns theta_y",
        ),
        pytest.param(
            lambda u_yx, theta_y: u_yx["u_y"] - theta_y,
            1.0,
            1.1,
            {},
            1e-12,
            id="r(x, z, l) = E[u_y] - theta_y",
        ),
        pytest.param(
            lambda u_yx, theta_y: (
                u_yx["x"] * jnp.exp(u_yx["u_y"] * theta_y[0]) + theta_y[1]
            ),
            jnp.array([1.0, 2.0]),
            jnp.array([2.0, 20.0]),
            {"learning_rate": 0.25},
            1e-4,
            id="r(x, z, l) = x e^{u_y * theta_y[0]} + theta_y[1]",
        ),
    ],
)
def test_learn_initialiser_ux_independent_regression_fn(
    jax_enable_x64,  # noqa: ARG001
    ricardo_regression_function,
    ux_independent_mlps,
    pytree_allclose,
    pytree_all_same_shape,
    f_y: MLPAlias,
    initial_theta_y_guess: ModelParam,
    expected_theta_y: ModelParam,
    opt_kwargs: dict | None,
    independent_param_atol: float,
    k_len: int = 5,
    z_len: int = 10,
    n_points: int = 1000,
    n_eval_pts: int = 50,
    independent_params: tuple[str, ...] = ("theta_m", "theta_r", "theta_pi"),
) -> None:
    r"""Test that `learn_optimiser` correctly minimises the function $B$ when provided a
    regression function $r = \mathbb{E}[f_Y(U, x, l; \theta_Y)]$.

    Various functional forms of $f_Y$ are tested. The 'analytic solution' that is used
    to compute the $\hat{r}_i$ points performs manual integration of $f_Y$ to determine
    the expected value.
    """
    if opt_kwargs is None:
        opt_kwargs = {}
    # Save on time when specifying parameters, by not having to include
    # independent parameters for the problem in the initial guess & expected
    # solution.
    expected_solution = {"theta_y": expected_theta_y}
    expected_solution.update(dict.fromkeys(independent_params, 1.0))
    initial_guess = {"theta_y": initial_theta_y_guess}
    initial_guess.update(dict.fromkeys(independent_params, 1.0))

    mlps, r_analytic = ux_independent_mlps(k_len, n_points, f_y)
    r = ricardo_regression_function(
        k_len=k_len,
        z_len=z_len,
        theta_x=0.0,
        n_points=n_points,
        **mlps,
    )

    evaluation_points = {
        "x": jnp.linspace(-1.0, 1.0, num=n_eval_pts),
        "z": jnp.arange(n_eval_pts),
        "l": 0.0,
    }
    evaluation_points_mapping = {"x": 0, "z": 0, "l": None}
    r_hat_pts = jax.vmap(r_analytic, in_axes=(evaluation_points_mapping, None))(
        evaluation_points, expected_solution
    )

    learn_initialiser = build_loss_function(
        r,
        evaluation_points,
        r_hat_pts,
        evaluation_points_axes_mapping=evaluation_points_mapping,
    )
    result = stochastic_gradient_descent(
        learn_initialiser,
        initial_guess,
        **opt_kwargs,
    )

    assert result.successful
    assert pytree_all_same_shape(result.fn_args, expected_solution)
    assert jnp.allclose(result.fn_args["theta_y"], expected_solution["theta_y"])
    # Other parameters may have moved around slightly due to autodiff imprecisions,
    # though in reality they should not have moved. Be gracious if necessary.
    assert pytree_allclose(
        {k: v for k, v in result.fn_args.items() if k in independent_params},
        {k: v for k, v in expected_solution.items() if k in independent_params},
        atol=independent_param_atol,
    )
