import re
from collections.abc import Callable
from typing import Any, Concatenate, TypeAlias

import jax
import numpy as np
import numpy.typing as npt
import numpyro
import pytest
from numpyro.distributions import Normal
from numpyro.infer import MCMC, NUTS

from causalprog.graph import DistributionNode, Graph, ParameterNode

# TODO: Refactor into conftest to share with test_node/test_create_model_site.py.
# Similarly for any other graphs here that are shared... and decorator-fixtures
MCMCRunner: TypeAlias = Callable[Concatenate[Callable, ...], MCMC]


@pytest.fixture(scope="session")
def mcmc_default_options() -> dict[str, float]:
    return {"num_warmup": 500, "num_samples": 1000}


@pytest.fixture
def run_nuts_mcmc(
    rng_key: jax.Array,
) -> MCMCRunner:
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
def two_normal_graph() -> Graph:
    g = Graph(label="Two Normals")

    mu_x = ParameterNode(label="mu_x")
    nu_y = ParameterNode(label="nu_y")

    x = DistributionNode(
        Normal,
        label="X",
        parameters={"loc": "mu_x"},
        constant_parameters={"scale": 1.0},
    )
    y = DistributionNode(
        Normal,
        label="Y",
        parameters={"loc": "X", "scale": "nu_y"},
    )
    g.add_node(mu_x)
    g.add_node(nu_y)
    g.add_edge(mu_x, x)
    g.add_edge(nu_y, y)
    g.add_edge(x, y)

    return g


@pytest.fixture
def two_normal_graph_expected_model() -> Callable[..., Callable[[], None]]:
    """Creates the model that the two_normal_graph should produce."""

    def _inner(mu_x: float, nu_y: float) -> Callable[[], None]:
        def _model():
            x = numpyro.sample("X", Normal(loc=mu_x, scale=1.0))
            numpyro.sample("Y", Normal(loc=x, scale=nu_y))

        return _model

    return _inner


def test_model_constructor(
    two_normal_graph: Graph,
    two_normal_graph_expected_model: Callable[[float, float], Callable[[], None]],
    run_nuts_mcmc: MCMCRunner,
    mcmc_default_options: dict[str, Any],
    collection_of_parameter_values: tuple[dict[str, npt.ArrayLike], ...] = (
        {"mu_x": 0.0, "nu_y": 1.0},
        {"mu_x": 1.0, "nu_y": 2.0},
    ),
) -> None:
    """Test the model_constructor.

    The `Graph.model_constructor` should only need to be invoked once per `Graph`.
    Though it does need to be "re-invoked" if the `Graph` is edited in any way after
    calling it previously.

    One the `model_constructor` has been invoked, it returns a function that should
    be able to construct a model represented by the `Graph`, _given_ values of the
    `ParameterNode`s to use. This means we should be able to invoke the output of
    `Graph.model_constructor` multiple times, with different sets of parameters, and
    each time produce a different realisation of the model.

    We check these properties in the following way:
    - First, we invoke `two_normal_graph.model_constructor()`, saving the result to
      the `constructor` variable.
    - Then, for each collection of parameters, we create a realisation from the
      `constructor`, by passing in that collection of parameters. We then confirm that
      this agrees with the explicit model that should have been constructed.
    """
    # The model builder only needs to be created once per Graph.
    constructor = two_normal_graph.model_constructor()
    assert callable(constructor)

    # Now, we should be able to use each set of parameters to create
    # realisations of the model.
    for param_values in collection_of_parameter_values:
        # Now realise the model with the parameter values we have given
        realisation = constructor(**param_values)
        assert callable(realisation)

        # And finally realise the expected model using the same parameter values
        expected_realisation = two_normal_graph_expected_model(**param_values)  # type: call_args

        # Confirm that the two models are indeed identical.
        # TODO: Refactor this into a "assert models are equal" method or something.
        via_model: dict[str, npt.ArrayLike] = run_nuts_mcmc(
            realisation,
            mcmc_kwargs=mcmc_default_options,
        ).get_samples()
        via_expected: dict[str, npt.ArrayLike] = run_nuts_mcmc(
            expected_realisation,
            mcmc_kwargs=mcmc_default_options,
        ).get_samples()

        for sample_name, samples in via_model.items():
            assert sample_name in via_expected
            assert np.allclose(samples, via_expected[sample_name])


def test_model_constructor_missing_parameter(
    two_normal_graph: Graph,
) -> None:
    """Any models build by a `Graph` will raise a `KeyError` when they are not provided
    values for all of the `ParameterNode`s, since this prevents the model from being
    realised.

    We can check for this behaviour by deliberately 'leaving out' one parameter from
    the `parameter_values` argument, and then attempting to invoke the model function
    that the `Graph` creates.
    """
    # Deliberately leave out the "nu_y" variable.
    parameter_values = {"mu_x": 0.0}
    # Which should result in the error below.
    expected_exception = KeyError("ParameterNode 'nu_y' not assigned")

    # Building the model itself should be OK, since this sets up the model function
    # which assumes it will be passed values for each parameter.
    constructor = two_normal_graph.model_constructor()

    # Attempting to construct a model with too few parameters
    # should now result in an error
    with pytest.raises(
        type(expected_exception), match=re.escape(str(expected_exception))
    ):
        constructor(**parameter_values)
