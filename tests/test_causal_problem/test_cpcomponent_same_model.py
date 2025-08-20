"""Tests for _CPComponent.can_use_same_model.

- This method should be symmetric in its arguments.
- Returns False when one of the arguments is not a _CPComponent instance.
- CausalEstimands and Constraints should still be able to share models.
- Models can be shared IFF the same handlers, in the same order, are applied.
"""

import pytest

from causalprog.causal_problem.causal_estimand import (
    CausalEstimand,
    Constraint,
    _CPComponent,
)
from causalprog.causal_problem.handlers import HandlerToApply


# HandlerToApply compares the handler argument with IS, so we need to instantiate here.
def handler_a(**pv) -> None:
    return


def handler_b(**pv) -> None:
    return


@pytest.mark.parametrize(
    ("component_1", "component_2", "expected_result"),
    [
        pytest.param(
            _CPComponent(
                HandlerToApply(handler_a, {}), do_with_samples=lambda **pv: None
            ),
            _CPComponent(
                HandlerToApply(handler_a, {}), do_with_samples=lambda **pv: None
            ),
            True,
            id="Same model as self",
        ),
        pytest.param(
            _CPComponent(do_with_samples=lambda **pv: None),
            _CPComponent(do_with_samples=lambda **pv: None),
            True,
            id="No effect handlers case is handled",
        ),
        pytest.param(
            _CPComponent(
                HandlerToApply(handler_a, {}), do_with_samples=lambda **pv: 1.0
            ),
            _CPComponent(
                HandlerToApply(handler_a, {}), do_with_samples=lambda **pv: 2.0
            ),
            True,
            id="_do_with_samples does not affect model compatibility",
        ),
        pytest.param(
            CausalEstimand(
                HandlerToApply(handler_a, {}), do_with_samples=lambda **pv: None
            ),
            Constraint(
                HandlerToApply(handler_a, {}), do_with_samples=lambda **pv: None
            ),
            True,
            id="CausalEstimand and Constraints can share models",
        ),
        pytest.param(
            _CPComponent(
                HandlerToApply(handler_a, {"option": "a"}),
                do_with_samples=lambda **pv: None,
            ),
            _CPComponent(
                HandlerToApply(handler_a, {"option": "b"}),
                do_with_samples=lambda **pv: None,
            ),
            False,
            id="Different handlers deny same model",
        ),
        pytest.param(
            _CPComponent(
                HandlerToApply(handler_a, {"option": "a"}),
                HandlerToApply(handler_a, {"option": "b"}),
                do_with_samples=lambda **pv: None,
            ),
            _CPComponent(
                HandlerToApply(handler_a, {"option": "b"}),
                HandlerToApply(handler_a, {"option": "a"}),
                do_with_samples=lambda **pv: None,
            ),
            False,
            id="Different handler order denies same model",
        ),
        pytest.param(
            _CPComponent(do_with_samples=lambda **pv: None),
            1.0,
            False,
            id="Compare to non-_CPComponent",
        ),
    ],
)
def test_can_use_same_model(
    component_1: _CPComponent, component_2: _CPComponent, *, expected_result: bool
) -> None:
    if isinstance(component_1, _CPComponent):
        assert component_1.can_use_same_model_as(component_2) == expected_result
    if isinstance(component_2, _CPComponent):
        assert component_2.can_use_same_model_as(component_1) == expected_result
