from collections.abc import Callable
from typing import Concatenate

import jax.numpy as jnp
import pytest


@pytest.fixture(scope="session")
def assert_within_mc_error() -> Callable[Concatenate[float, float, int, ...], None]:
    def _assert_within_mc_error(
        x: float, y: float, n_samples: int, forgiveness_factor: float = 1.25
    ) -> None:
        """
        Shortcut function for testing computed values of MC integrals.

        MC integrals are inherently stochastic, but in general we expect that the
        absolute error (of the computed integral from the true value) decreases
        as roughly the square-root of the number of samples. The `forgiveness_factor`
        is essentially a quick-hack to get around the fact that the shape of the
        integrand also affects the quality of the approximation, and to provide us with
        some wiggle-room.
        """
        assert jnp.abs(x - y) <= forgiveness_factor / jnp.sqrt(n_samples)

    return _assert_within_mc_error
