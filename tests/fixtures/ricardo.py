from collections.abc import Callable
from typing import TypeAlias

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
