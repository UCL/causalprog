from collections.abc import Callable

import numpy.typing as npt
import numpyro
import pytest
from numpyro.distributions import Normal

from causalprog.graph.node import DistributionNode


@pytest.mark.parametrize(
    ("node", "dependent_nodes", "identical_model"),
    [
        pytest.param(
            DistributionNode(
                Normal,
                label="Normal",
                constant_parameters={"loc": 10.0, "scale": 10.0},
            ),
            # This is not a useless lambda, we need to delay evaluation until inside the
            # model function.
            lambda: {},  # noqa: PIE807
            lambda: numpyro.sample("Normal", Normal(loc=10.0, scale=10.0)),
            id="Constant parameters only",
        ),
        pytest.param(
            DistributionNode(
                Normal,
                label="Normal",
                parameters={"loc": "mu"},
                constant_parameters={"scale": 1.0},
            ),
            lambda: {"mu": 0.0},
            lambda: numpyro.sample("Normal", Normal(loc=0.0, scale=1.0)),
            id="Parameter node dependency",
        ),
        pytest.param(
            DistributionNode(
                Normal,
                label="Normal",
                parameters={"loc": "mu"},
                constant_parameters={"scale": 1.0},
            ),
            lambda: {"mu": numpyro.sample("mu", Normal(0.0, 1.0))},
            lambda: numpyro.sample(
                "Normal",
                Normal(loc=numpyro.sample("mu", Normal(0.0, 1.0)), scale=1.0),
            ),
            id="DistributionNode dependency",
        ),
        pytest.param(
            DistributionNode(
                Normal,
                label="Normal",
                constant_parameters={"loc": 0.0, "scale": 1.0},
            ),
            lambda: {"tau": 1.0},
            lambda: numpyro.sample("Normal", Normal(loc=0.0, scale=1.0)),
            id="Un-needed nodes are ignored",
        ),
        pytest.param(
            DistributionNode(
                Normal,
                label="Normal",
                parameters={"loc": "mu"},
                constant_parameters={"scale": 1.0},
            ),
            lambda: {"not_mu": 1.0},
            KeyError("mu"),
            id="Missing dependency",
        ),
    ],
)
def test_create_model_site(
    node: DistributionNode,
    dependent_nodes: Callable[[], dict[str, npt.ArrayLike]],
    identical_model: Exception | Callable[[], npt.ArrayLike],
    assert_samples_are_identical,
    raises_context,
    run_nuts_mcmc,
    mcmc_default_options: dict[str, float],
) -> None:
    """Test use and error cases for create_distribution."""
    if isinstance(identical_model, Exception):
        with raises_context(identical_model):
            node.create_model_site(**dependent_nodes())
    else:
        via_method = run_nuts_mcmc(
            lambda: node.create_model_site(**dependent_nodes()),
            mcmc_kwargs=mcmc_default_options,
        )
        trusted = run_nuts_mcmc(
            identical_model,
            mcmc_kwargs=mcmc_default_options,
        )

        assert_samples_are_identical(via_method, trusted)
