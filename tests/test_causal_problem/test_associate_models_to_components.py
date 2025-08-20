"""Test the association of models to the components of the CausalProblem.

One model is needed fr each unique combination of handlers that the Constraints and
CausalEstimand possess. We can mimic this behaviour by defining a single handler, and
then passing different numbers of copies of this handler in to our Constraints and
CausalEstimand. Different numbers of handlers force different models, and thus we should
end up with one model for each unique number of copies that we use.

Components are examined in the `_ordered_components` order, which goes through the
`constraints` list first, and then the CausalEstimand at the end. As such, models are
also "created" in this order.
"""

from collections.abc import Callable

import pytest

from causalprog.causal_problem import (
    CausalEstimand,
    CausalProblem,
    Constraint,
    HandlerToApply,
)
from causalprog.graph import Graph, ParameterNode


@pytest.fixture
def underlying_graph() -> Graph:
    """The underlying graph is not actually important for checking model association,
    so we just return a single node graph.
    """
    g = Graph(label="Placeholder")
    g.add_node(ParameterNode(label="p"))
    return g


def placeholder_handler_fn(*args, **kwargs) -> None:
    """IS comparison means this function does need to be statically defined.

    args[0] is the model input, so we just effectively return the same model.
    """
    return args[0]


@pytest.fixture
def placeholder_handler() -> Callable[[], HandlerToApply]:
    """Creates a HandlerToApply instance.

    We will use copies of the returned instance to "trick" the CasualProblem
    class into creating additional models due to different numbers of handlers
    being applied to its components.
    """

    def _inner() -> HandlerToApply:
        return HandlerToApply(placeholder_handler_fn, {})

    return _inner


@pytest.mark.parametrize(
    (
        "handlers_to_give_to_constraints",
        "handlers_to_give_estimand",
        "expected_components_to_models_mapping",
    ),
    [
        # There are no constraints, so the CausalEstimand uses the only model created.
        pytest.param([], 0, [0], id="No constraints"),
        # All constraints and the causal estimand use the same model.
        pytest.param([0] * 3, 0, [0] * 4, id="Same model used by all"),
        # 1st constraint: Model w/ 1 handler created, taking model index 0.
        # 2nd constraint: re-uses model index 0 (with 1 handler).
        # CE: Model w/ 0 handlers created, taking model index 1.
        pytest.param([1] * 2, 0, [0, 0, 1], id="CausalEstimand is always last model"),
        # 1st constraint: Model w/ 1 handler created, taking model index 0.
        # 2nd constraint: Model w/ 0 handlers created, taking model index 1.
        # 3rd constraint: re-uses model index 0 (with 1 handler).
        # 4th constraint: Model w/ 2 handlers created, taking model index 2.
        # CE: re-uses model index 0 (with 1 handler).
        pytest.param(
            [1, 0, 1, 2], 1, [0, 1, 0, 2, 0], id="Models created in particular order"
        ),
    ],
)
def test_associate_models_to_components(
    handlers_to_give_to_constraints: list[int],
    handlers_to_give_estimand: int,
    expected_components_to_models_mapping: list[int],
    placeholder_handler: Callable[[], HandlerToApply],
    underlying_graph: Graph,
    n_samples: int = 1,
) -> None:
    # The number of models is the number of 'unique numbers of handlers' given to the
    # constraints and causal estimand.
    expected_number_of_models = len(
        set(handlers_to_give_to_constraints).union({handlers_to_give_estimand})
    )

    constraints = [
        Constraint(
            *(placeholder_handler() for _ in range(copies)),
            do_with_samples=lambda **pv: None,
        )
        for copies in handlers_to_give_to_constraints
    ]
    ce = CausalEstimand(
        *(placeholder_handler() for _ in range(handlers_to_give_estimand)),
        do_with_samples=lambda **pv: None,
    )

    cp = CausalProblem(underlying_graph, *constraints, causal_estimand=ce)
    models, association = cp._associate_models_to_components(n_samples)  # noqa: SLF001

    assert len(models) == expected_number_of_models
    assert association == expected_components_to_models_mapping
