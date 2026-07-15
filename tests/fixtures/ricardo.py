from collections.abc import Callable
from typing import TypeAlias

import jax.numpy as jnp
import pytest

from causalprog.graph.ricardo import (
    MLPAlias,
    ModelParam,
    build_regression_function,
    example_model,
)
from causalprog.quadrature import UniformWeightMonteCarloGaussianQuadrature as UWMCGQuad

RegressionBuilder: TypeAlias = Callable[
    [
        int,
        int,
        MLPAlias,
        MLPAlias,
        MLPAlias,
        MLPAlias,
        MLPAlias,
        ModelParam,
        int,
    ],
    MLPAlias,
]


@pytest.fixture
def ricardo_regression_function(rng_key) -> RegressionBuilder:
    def _inner(
        k_len: int,
        z_len: int,
        f_ux: MLPAlias,
        f_pi: MLPAlias,
        f_y: MLPAlias,
        f_r: MLPAlias,
        f_m: MLPAlias,
        theta_x: ModelParam,
        n_points: int,
    ):
        """Fast assembly of an appropriate regression function, given necessary inputs.

        Internal testing use only.
        Refactored to help separate test steps and test setup.
        """
        g = example_model(
            k=k_len,
            z_len=z_len,
            compute_u_x=f_ux,
            compute_u_y=f_pi,
            compute_phi_x=None,
            compute_x=None,
            compute_y=f_y,
        )
        # Manually attach methods to node for now. FIXME: should be removed once we have
        # a more elegant solution for attaching additional functions to nodes.
        g.get_node("u_y").f_r = f_r
        g.get_node("u_y").f_m = f_m

        return build_regression_function(
            g, theta_x, UWMCGQuad(n_points, rng_key=rng_key)
        )

    return _inner


@pytest.fixture
def uy_independent_mlps() -> Callable[[int], tuple[dict[str, MLPAlias], MLPAlias]]:
    """Return `MLPAliases` that result in a $U_Y$-independent regression function
    (in a `dict` with `str` keys), and the expected analytic function.

    The returned lambda functions essentially set
    $f_Y(u_y, x, l; \theta_Y) = \theta_Y[0]e^{-x^2}$, when using Ricardo's graph setup.

    Note that `f_r`, `f_m`, `f_pi`, and `f_ux` are given be non-trivial (but also
    unrealistic) forms, just so that we can check the auto-diff isn't making too much of
    a mess when we take derivatives of variables that (in theory) the regression
    function $r$ doesn't depend on.

    `k_len` is required to ensure that `f_pi` has the correct shape, so must be passed
    as an argument.
    """

    def _inner(k_len: int) -> tuple[dict[str, MLPAlias], MLPAlias]:
        return {
            "f_r": lambda czl, theta_m: theta_m * jnp.ones_like(czl["z"]),
            "f_m": lambda czl, theta_r: czl["c"] * theta_r,
            "f_ux": lambda xzl, theta_x: xzl["x"] * theta_x[0],
            "f_pi": lambda ucl, theta_pi: (
                theta_pi * ucl["c"] + ucl["u_x"] * jnp.arange(k_len)
            ),
            "f_y": lambda xu_y, theta_y: theta_y[0] * jnp.exp(-(xu_y["x"] ** 2)),
        }, lambda xzl, theta: theta["theta_y"][0] * jnp.exp(-(xzl["x"] ** 2))

    return _inner
