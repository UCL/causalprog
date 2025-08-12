import numpy.typing as npt
import numpyro
import pytest

from causalprog.graph import DistributionNode, Graph, ParameterNode


@pytest.mark.parametrize(
    "param_values",
    [
        pytest.param({"mu_x": 0.0, "nu_y": 1.0}, id="mean(X) = 0, cov(UX) = 1"),
        pytest.param({"mu_x": 1.0, "nu_y": 2.0}, id="mean(X) = 1, cov(UX) = 2"),
    ],
)
def test_model(
    param_values: dict[str, npt.ArrayLike],
    two_normal_graph_parametrized_mean,
    two_normal_graph_expected_model,
    assert_samples_are_identical,
    run_default_nuts_mcmc,
) -> None:
    """Test the `Graph.model` method.

    `Graph.model` takes values for the `ParameterNode`s (parameters of the model)
    as its arguments. It is designed to be able to be used just like any other
    function defining a model, namely that `Graph.model(**parameter_values)`
    is a function that creates the appropriate model sites, given values for the
    model parameters.

    As such, we can check the model is constructed correctly by comparing an
    MCMC sampling output of `Graph.model` with the explicit model that it should
    correspond to.
    """
    graph = two_normal_graph_parametrized_mean()
    assert callable(graph.model)

    via_model = run_default_nuts_mcmc(
        graph.model,
        mcmc_run_kwargs=param_values,
    )
    via_expected = run_default_nuts_mcmc(
        two_normal_graph_expected_model,
        mcmc_run_kwargs=param_values,
    )

    assert_samples_are_identical(via_model, via_expected)


def test_model_missing_parameter(
    two_normal_graph_parametrized_mean,
    raises_context,
    seed: int,
) -> None:
    """`Graph.model` will raise a `KeyError` when a value is not passed for
    a `ParameterNode`.
    """
    graph = two_normal_graph_parametrized_mean()

    # Deliberately leave out the "nu_y" variable.
    parameter_values = {"mu_x": 0.0}
    # Which should result in the error below.
    expected_exception = KeyError("ParameterNode 'nu_y' not assigned")

    # Not passing enough parameters should be picked up by the model.
    with raises_context(expected_exception), numpyro.handlers.seed(rng_seed=seed):
        graph.model(**parameter_values)


def test_model_extension(
    two_normal_graph_parametrized_mean,
    assert_samples_are_identical,
    run_default_nuts_mcmc,
) -> None:
    """Test that `Graph.model` can be extended."""
    graph = two_normal_graph_parametrized_mean()

    parameter_values = {"mu_x": 0.0, "nu_y": 1.0}

    # Build the graph, but without the X-node.
    mu_x = ParameterNode(label="mu_x")
    x = DistributionNode(
        numpyro.distributions.Normal,
        label="UX",
        parameters={"loc": "mu_x"},
        constant_parameters={"scale": 1.0},
    )
    one_normal_graph = Graph(label="One normal")
    one_normal_graph.add_edge(mu_x, x)

    def extended_model(*, nu_y, **parameter_values):
        sites = one_normal_graph.model(**parameter_values)
        numpyro.sample(
            "X",
            numpyro.distributions.Normal(
                loc=sites["UX"],
                scale=nu_y,
            ),
        )

    via_explicit_graph = run_default_nuts_mcmc(
        graph.model, mcmc_run_kwargs=parameter_values
    )
    via_extended = run_default_nuts_mcmc(
        extended_model, mcmc_run_kwargs=parameter_values
    )

    assert_samples_are_identical(via_explicit_graph, via_extended)
