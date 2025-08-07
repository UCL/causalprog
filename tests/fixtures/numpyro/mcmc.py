"""Helper functions for running MCMCs on numpyro models."""

from collections.abc import Callable
from typing import Concatenate, TypeAlias

import pytest
from jax import Array
from numpyro.infer import MCMC, NUTS

MCMCRunner: TypeAlias = Callable[Concatenate[Callable, ...], MCMC]


@pytest.fixture(scope="session")
def mcmc_default_options() -> dict[str, float]:
    """Default options used when running a numpyro.infer MCMC."""
    return {"num_warmup": 500, "num_samples": 1000}


@pytest.fixture
def run_nuts_mcmc(
    rng_key: Array,
) -> MCMCRunner:
    """Run an MCMC using a NUTS (No-U-Turns) method.

    run_nuts_mcmc(model) returns the MCMC instance that would result from
    setting up a NUTS kernel and passing it to numpyro.infer.MCMC.
    """

    def inner(model, *, nuts_kwargs=None, mcmc_kwargs=None) -> MCMC:
        if not nuts_kwargs:
            nuts_kwargs = {}
        if not mcmc_kwargs:
            mcmc_kwargs = {}

        kernel = NUTS(model, **nuts_kwargs)
        mcmc = MCMC(kernel, **mcmc_kwargs)
        mcmc.run(rng_key)
        return mcmc

    return inner


@pytest.fixture
def run_default_nuts_mcmc(
    mcmc_default_options: dict[str, float], run_nuts_mcmc: MCMCRunner
) -> MCMCRunner:
    """Run an MCMC using the default options (for tests within the test suite)."""

    def inner(model) -> MCMC:
        return run_nuts_mcmc(model, mcmc_kwargs=mcmc_default_options)

    return inner
