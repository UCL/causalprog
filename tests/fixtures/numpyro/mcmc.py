"""Helper functions for running MCMCs on numpyro models."""

from collections.abc import Callable
from typing import Concatenate, TypeAlias

import numpy as np
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

    def inner(
        model, *, nuts_kwargs=None, mcmc_kwargs=None, mcmc_run_kwargs=None
    ) -> MCMC:
        if not nuts_kwargs:
            nuts_kwargs = {}
        if not mcmc_kwargs:
            mcmc_kwargs = {}
        if not mcmc_run_kwargs:
            mcmc_run_kwargs = {}

        kernel = NUTS(model, **nuts_kwargs)
        mcmc = MCMC(kernel, **mcmc_kwargs)
        mcmc.run(rng_key, **mcmc_run_kwargs)
        return mcmc

    return inner


@pytest.fixture
def run_default_nuts_mcmc(
    mcmc_default_options: dict[str, float], run_nuts_mcmc: MCMCRunner
) -> MCMCRunner:
    """Run an MCMC using the default options (for tests within the test suite)."""

    def inner(model, *, mcmc_run_kwargs=None) -> MCMC:
        return run_nuts_mcmc(
            model, mcmc_kwargs=mcmc_default_options, mcmc_run_kwargs=mcmc_run_kwargs
        )

    return inner


@pytest.fixture(scope="session")
def assert_samples_are_identical() -> Callable[[MCMC, MCMC], None]:
    """Assert that samples produced by two MCMC instances are identical.

    `True` is returned if the MCMC instances have the same sample names,
    and the samples associated with each sample name match (as per `numpy.allclose`).
    """

    def _inner(left_mcmc: MCMC, right_mcmc: MCMC) -> None:
        samples_l: dict[str, Array] = left_mcmc.get_samples()
        samples_r: dict[str, Array] = right_mcmc.get_samples()

        for sample_name, sample_values in samples_l.items():
            # Confirm samples on right are contained in samples on left.
            assert sample_name in samples_r, (
                f"Samples on left ({sample_name}) not present on right"
            )
            # Confirm samples match.
            assert np.allclose(sample_values, samples_r[sample_name]), (
                f"Samples '{sample_name}' do not match"
            )
        for sample_name in samples_r:
            # Confirm samples on left are contained in samples on right.
            # Combined with the above, this also implies the names of the
            # samples are identical, ERGO there are no more sets of samples
            # to compare (as they were checked in the previous for-loop).
            assert sample_name in samples_l, (
                f"Samples on right ({sample_name}) not present on left"
            )

    return _inner
