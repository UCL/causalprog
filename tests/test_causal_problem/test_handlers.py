import pytest

from causalprog.causal_problem import HandlerToApply
from causalprog.causal_problem._base_component import EffectHandler


def placeholder_callable() -> EffectHandler:
    """Stand-in for an effect handler."""
    return lambda model, **kwargs: (lambda **pv: model(**kwargs))


@pytest.mark.parametrize(
    (
        "args",
        "expected_error",
        "use_classmethod",
    ),
    [
        pytest.param(
            (placeholder_callable, {}), None, False, id="Standard construction"
        ),
        pytest.param((placeholder_callable, {}), None, True, id="Via from_pair"),
        pytest.param(
            ({}, placeholder_callable), None, True, id="Via from_pair (out of order)"
        ),
        pytest.param(
            (placeholder_callable, []),
            TypeError(
                "Options should be dictionary mapping option arguments to values "
                "(got list)."
            ),
            False,
            id="Wrong options type",
        ),
        pytest.param(
            (1.0, {}),
            TypeError("float is not callable."),
            False,
            id="Handler is not callable",
        ),
        pytest.param(
            (0, 0, 0),
            ValueError(
                "HandlerToApply can only be constructed from a container of 2 elements"
            ),
            True,
            id="Tuple too long",
        ),
    ],
)
def test_handlertoapply_creation(
    args: tuple[EffectHandler | dict, ...],
    expected_error: Exception | None,
    raises_context,
    *,
    use_classmethod: bool,
):
    if isinstance(expected_error, Exception):
        if use_classmethod:
            with raises_context(expected_error):
                HandlerToApply.from_pair(args)
        else:
            with raises_context(expected_error):
                HandlerToApply(*args)
    else:
        handler = (
            HandlerToApply.from_pair(args) if use_classmethod else HandlerToApply(*args)
        )

        assert isinstance(handler.options, dict)
        assert callable(handler.handler)


@pytest.mark.parametrize(
    ("left", "right", "expected_result"),
    [
        pytest.param(
            HandlerToApply(placeholder_callable, {"option1": 1.0}),
            HandlerToApply(placeholder_callable, {"option1": 1.0}),
            True,
            id="Identical",
        ),
        pytest.param(
            HandlerToApply(lambda **pv: None, {"option1": 1.0}),
            HandlerToApply(lambda **pv: None, {"option1": 1.0}),
            False,
            id="callables compared using IS",
        ),
        pytest.param(
            HandlerToApply(placeholder_callable, {"option1": 1.0}),
            HandlerToApply(placeholder_callable, {"option2": 1.0}),
            False,
            id="Options must match",
        ),
        pytest.param(
            HandlerToApply(placeholder_callable, {"option1": 1.0}),
            1.0,
            False,
            id="Comparison to different object class",
        ),
    ],
)
def test_handlertoapply_equality(
    left: object, right: object, *, expected_result: bool
) -> None:
    assert (left == right) == expected_result
    assert (left == right) == (right == left)
