import re
from inspect import Parameter, Signature

import pytest

from causalprog.backend._convert_signature import _signature_can_be_cast
from causalprog.backend._typing import ParamNameMap, StaticValues


@pytest.mark.parametrize(
    (
        "signature_to_convert",
        "new_signature",
        "old_to_new_names",
        "give_static_value",
        "expected_output",
    ),
    [
        pytest.param(
            Signature(
                [
                    Parameter("a", Parameter.POSITIONAL_ONLY),
                    Parameter("b", Parameter.POSITIONAL_ONLY),
                ]
            ),
            Signature(
                [
                    Parameter("a", Parameter.POSITIONAL_ONLY),
                ]
            ),
            {},
            {},
            ValueError(
                "Parameter 'b' has no counterpart in new_signature, "
                "and does not take a static value."
            ),
            id="Parameter not matched.",
        ),
        pytest.param(
            Signature(
                [
                    Parameter("a", Parameter.POSITIONAL_ONLY),
                    Parameter("b", Parameter.POSITIONAL_ONLY),
                ]
            ),
            Signature(
                [
                    Parameter("a", Parameter.POSITIONAL_ONLY),
                ]
            ),
            {"a": "a", "b": "a"},
            {},
            ValueError("Parameter 'a' is mapped to by multiple parameters."),
            id="Two arguments mapped to a single parameter.",
        ),
        pytest.param(
            Signature(
                [
                    Parameter("vargs", Parameter.VAR_POSITIONAL),
                ]
            ),
            Signature(
                [
                    Parameter("a", Parameter.POSITIONAL_ONLY),
                ]
            ),
            {"vargs": "a"},
            {},
            ValueError(
                "Variable-length positional/keyword parameters must map to each other "
                "('vargs' is type VAR_POSITIONAL, but 'a' is type POSITIONAL_ONLY)."
            ),
            id="Map *args to positional argument.",
        ),
        pytest.param(
            Signature(
                [
                    Parameter("vargs", Parameter.VAR_POSITIONAL),
                ]
            ),
            Signature(
                [
                    Parameter("kwarg", Parameter.VAR_KEYWORD),
                ]
            ),
            {"vargs": "kwarg"},
            {},
            ValueError(
                "Variable-length positional/keyword parameters must map to each other "
                "('vargs' is type VAR_POSITIONAL, but 'kwarg' is type VAR_KEYWORD)."
            ),
            id="Map *args to **kwargs.",
        ),
        pytest.param(
            Signature(
                [
                    Parameter("a", Parameter.POSITIONAL_ONLY),
                ]
            ),
            Signature(
                [
                    Parameter("a", Parameter.POSITIONAL_ONLY),
                    Parameter("b", Parameter.POSITIONAL_ONLY),
                ]
            ),
            {},
            {},
            ValueError("Some parameters in new_signature are not used: b"),
            id="new_signature contains extra parameters.",
        ),
        pytest.param(
            "general_function_signature",
            "general_function_signature",
            {},
            {},
            (
                {
                    "posix": "posix",
                    "posix_def": "posix_def",
                    "vargs": "vargs",
                    "kwo": "kwo",
                    "kwo_def": "kwo_def",
                    "kwargs": "kwargs",
                },
                {},
            ),
            id="Can cast to yourself.",
        ),
        pytest.param(
            Signature([Parameter("a", Parameter.POSITIONAL_ONLY)]),
            Signature([Parameter("a", Parameter.KEYWORD_ONLY)]),
            {},
            {},
            ({"a": "a"}, {}),
            id="Infer identically named parameter (even with type change)",
        ),
        pytest.param(
            Signature([Parameter("args", Parameter.VAR_POSITIONAL)]),
            Signature([Parameter("new_args", Parameter.VAR_POSITIONAL)]),
            {},
            {},
            ({"args": "new_args"}, {}),
            id="Infer VAR_POSITIONAL matching.",
        ),
        pytest.param(
            Signature([Parameter("a", Parameter.POSITIONAL_ONLY)]),
            Signature([]),
            {},
            {"a": 10},
            ({}, {"a": 10}),
            id="Assign static value to argument without default.",
        ),
        pytest.param(
            Signature([Parameter("a", Parameter.POSITIONAL_ONLY, default=10)]),
            Signature([]),
            {},
            {},
            ({}, {"a": 10}),
            id="Infer static value from argument default.",
        ),
    ],
)
def test_signature_can_be_cast(
    signature_to_convert: Signature,
    new_signature: Signature,
    old_to_new_names: ParamNameMap,
    give_static_value: StaticValues,
    expected_output: Exception | tuple[str | None, ParamNameMap, StaticValues],
    request,
) -> None:
    if isinstance(signature_to_convert, str):
        signature_to_convert = request.getfixturevalue(signature_to_convert)
    if isinstance(new_signature, str):
        new_signature = request.getfixturevalue(new_signature)

    if isinstance(expected_output, Exception):
        with pytest.raises(
            type(expected_output), match=re.escape(str(expected_output))
        ):
            _signature_can_be_cast(
                signature_to_convert,
                new_signature,
                old_to_new_names,
                give_static_value,
            )
    else:
        computed_output = _signature_can_be_cast(
            signature_to_convert,
            new_signature,
            old_to_new_names,
            give_static_value,
        )

        assert computed_output == expected_output
