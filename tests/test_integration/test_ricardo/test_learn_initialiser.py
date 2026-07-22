import jax
import jax.numpy as jnp
import pytest

from causalprog.graph.ricardo import MLPAlias, ModelParam, build_learn_initialiser
from causalprog.solvers.sgd import stochastic_gradient_descent


@pytest.mark.parametrize(
    ("opt_args", "opt_kwargs", "expected_solution"),
    [
        pytest.param(
            ({"a": 1.5, "b": 0.5, "c": -0.5},),
            {
                "convergence_criteria": lambda x, _: jnp.sqrt(jnp.abs(x)),
                "tolerance": 1e-6,
            },
            {"a": 2.0, "b": 1.0, "c": -1.0},
            id="SDG: Solution basin 1",
        ),
        pytest.param(
            ({"a": 0.5, "b": 1.5, "c": -0.5},),
            {
                "convergence_criteria": lambda x, _: jnp.sqrt(jnp.abs(x)),
                "tolerance": 1e-6,
            },
            {"a": 1.0, "b": 2.0, "c": -1.0},
            id="SDG: Solution basin 2 (switch initial guess params)",
        ),
    ],
)
def test_learn_initialiser_deterministic_fn(
    opt_args, opt_kwargs, expected_solution, pytree_allclose
) -> None:
    """Test that `learn_initialiser` correctly 'learns' the parameters of a
    deterministic function.
    """
    theta_opt = {"a": 1.0, "b": 2.0, "c": -1.0}

    def _r(data, theta):
        return (
            (data["x"] - theta["a"])
            * (data["x"] - theta["b"])
            * (data["x"] - theta["c"])
        )

    eval_pts = {"x": jnp.linspace(-5.0, 5.0, num=25)}
    r_hat_pts = _r(eval_pts, theta_opt)

    learn_initialiser = build_learn_initialiser(_r, eval_pts, r_hat_pts)
    result = stochastic_gradient_descent(learn_initialiser, *opt_args, **opt_kwargs)

    assert result.successful
    assert pytree_allclose(result.fn_args, expected_solution)


def _n_eval() -> int:
    """Magic number function for the number of evaluation points in tests.

    `pytest` fixtures cannot be evaluated as part of a parameter value. As such,
    in order to avoid magic numbers everywhere, we have a hidden function to
    fix a value for the number of evaluation points to use.
    """
    return 5


@pytest.mark.parametrize(
    (
        "eval_pts",
        "eval_pts_mapping",
        "r_hat_pts",
        "expected_solution",
    ),
    [
        pytest.param(
            {"x": jnp.linspace(-1.0, 1.0, num=_n_eval())},
            None,
            jnp.zeros(_n_eval()),
            {"a": 0.0, "b": 0.0},
            id="Standard scalar inputs.",
        ),
        pytest.param(
            {"x": jnp.tile(jnp.linspace(-1.0, 1.0, num=_n_eval()), (3, 1))},
            {"x": 1},
            jnp.ones(_n_eval()),
            {"a": 1.0, "b": jnp.zeros(3)},
            id="Vector-valued data",
        ),
        pytest.param(
            {"x": jnp.tile(jnp.linspace(-1.0, 1.0, num=_n_eval()), (3, 1)).T},
            {"x": 0},
            jnp.ones(_n_eval()),
            {"a": 1.0, "b": jnp.zeros(3)},
            id="Vector-valued data, transposed",
        ),
        pytest.param(
            {"x": jnp.tile(jnp.linspace(-1.0, 1.0, num=_n_eval()), (3, 4, 1))},
            {"x": 2},
            jnp.ones(_n_eval()),
            {"a": 1.0, "b": jnp.zeros((3, 4))},
            id="Matrix-valued data",
        ),
        pytest.param(
            {"x": jnp.tile(jnp.linspace(-1.0, 1.0, num=_n_eval()), (3, 1))},
            {"x": 1},
            jnp.array([6.0, 9.0 / 4.0, 1.0, 9.0 / 4.0, 6.0]),
            {"a": 1.0, "b": jnp.array([2.0, 1.0, 0.0])},
            id="Non-trivial problem",
        ),
    ],
)
def test_learn_initialiser_evaluation_points_axes_mapping(
    eval_pts: dict[str, jax.Array],
    eval_pts_mapping: dict[str, int],
    r_hat_pts: jax.Array,
    expected_solution: dict[str, jax.Array | float],
    pytree_allclose,
    pytree_all_same_shape,
) -> None:
    """Check that `evaluation_points_axes_mapping` is respected.

    This is tested by passing in data of various sizes, and confirming that the
    optimisation still runs and the resulting output has the expected shape for
    the $\theta$ parameters.
    """

    def _r(data, theta):
        return ((theta["b"] * data["x"]) ** 2).sum() + theta["a"]

    # Since we're only checking array dimension matching,
    # start the solver at the solution to immediately terminate.
    learn_initialiser = build_learn_initialiser(
        _r,
        eval_pts,
        r_hat_pts,
        evaluation_points_axes_mapping=eval_pts_mapping,
    )
    result = stochastic_gradient_descent(learn_initialiser, expected_solution)

    assert result.successful
    assert pytree_all_same_shape(result.fn_args, expected_solution)
    assert pytree_allclose(result.fn_args, expected_solution)


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
    learn_initialiser = build_learn_initialiser(
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

    learn_initialiser = build_learn_initialiser(
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
