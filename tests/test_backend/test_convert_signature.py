import re
from inspect import Parameter, Signature, signature

import pytest

from causalprog.backend._convert_signature import (
    _signature_can_be_cast,
    _validate_variable_length_parameters,
)
from causalprog.backend._typing import ParamNameMap, StaticValues


def general_function(posix, posix_def=0, *vargs, kwo, kwo_def=0, **kwargs):
    return posix, posix_def, vargs, kwo, kwo_def, kwargs


@pytest.mark.parametrize(
    ("signature", "exception"),
    [
        pytest.param(
            Signature(
                (
                    Parameter("vargs1", Parameter.VAR_POSITIONAL),
                    Parameter("vargs2", Parameter.VAR_POSITIONAL),
                )
            ),
            ValueError("New signature takes more than 1 VAR_POSITIONAL argument."),
            id="Two variable-length positional arguments.",
        ),
        pytest.param(
            Signature(
                (
                    Parameter("kwargs1", Parameter.VAR_KEYWORD),
                    Parameter("kwargs2", Parameter.VAR_KEYWORD),
                )
            ),
            ValueError("New signature takes more than 1 VAR_KEYWORD argument."),
            id="Two variable-length keyword arguments.",
        ),
        pytest.param(
            signature(general_function), None, id="Valid, but complex, signature."
        ),
        pytest.param(
            Signature(
                (
                    Parameter("arg1", Parameter.POSITIONAL_OR_KEYWORD),
                    Parameter("arg2", Parameter.POSITIONAL_OR_KEYWORD, default=1),
                    Parameter("vargs1", Parameter.VAR_POSITIONAL),
                    Parameter("vargs2", Parameter.VAR_POSITIONAL),
                    Parameter("kwargs1", Parameter.VAR_KEYWORD),
                )
            ),
            ValueError("New signature takes more than 1 VAR_POSITIONAL argument."),
            id="Two variable-length positional arguments, mixed with others.",
        ),
    ],
)
def test_validate_variable_length_parameters(
    signature: Signature, exception: Exception | None
):
    if exception is not None:
        with pytest.raises(type(exception), match=re.escape(str(exception))):
            _validate_variable_length_parameters(signature)
    else:
        _validate_variable_length_parameters(signature)


@pytest.mark.parametrize(
    (
        "signature_to_convert",
        "new_signature",
        "param_name_map",
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
                "Parameter 'b' has no counterpart in new_signature, and does not take a static value."
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
                    Parameter("varg", Parameter.VAR_POSITIONAL),
                ]
            ),
            Signature(
                [
                    Parameter("a", Parameter.POSITIONAL_ONLY),
                ]
            ),
            {"varg": "a"},
            {},
            ValueError(
                "Variable-length positional/keyword parameters must map to each other ('varg' is type VAR_POSITIONAL, but 'a' is type POSITIONAL_ONLY)."
            ),
            id="Map *args to positional argument.",
        ),
        pytest.param(
            Signature(
                [
                    Parameter("varg", Parameter.VAR_POSITIONAL),
                ]
            ),
            Signature(
                [
                    Parameter("kwarg", Parameter.VAR_KEYWORD),
                ]
            ),
            {"varg": "kwarg"},
            {},
            ValueError(
                "Variable-length positional/keyword parameters must map to each other ('varg' is type VAR_POSITIONAL, but 'kwarg' is type VAR_KEYWORD)."
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
            signature(general_function),
            signature(general_function),
            {},
            {},
            ("vargs", {key: key for key in signature(general_function).parameters}, {}),
            id="Can cast to yourself.",
        ),
        pytest.param(
            Signature([Parameter("a", Parameter.POSITIONAL_ONLY)]),
            Signature([Parameter("a", Parameter.KEYWORD_ONLY)]),
            {},
            {},
            (None, {"a": "a"}, {}),
            id="Infer identically named parameter (even with type change)",
        ),
        pytest.param(
            Signature([Parameter("args", Parameter.VAR_POSITIONAL)]),
            Signature([Parameter("new_args", Parameter.VAR_POSITIONAL)]),
            {},
            {},
            (None, {"args": "new_args"}, {}),
            id="Infer VAR_POSITIONAL matching.",
        ),
    ],
)
def test_signature_can_be_cast(
    signature_to_convert: Signature,
    new_signature: Signature,
    param_name_map: ParamNameMap,
    give_static_value: StaticValues,
    expected_output: Exception | tuple[str | None, ParamNameMap, StaticValues],
) -> None:
    if isinstance(expected_output, Exception):
        with pytest.raises(
            type(expected_output), match=re.escape(str(expected_output))
        ):
            _signature_can_be_cast(
                signature_to_convert,
                new_signature,
                param_name_map,
                give_static_value,
            )
    else:
        expected_old_varg = expected_output[0]
        expected_param_map = expected_output[1]
        expected_static_values = expected_output[2]

        old_varg, filled_param_map, filled_static_values = _signature_can_be_cast(
            signature_to_convert,
            new_signature,
            param_name_map,
            give_static_value,
        )

        assert old_varg == expected_old_varg
        assert filled_param_map == expected_param_map
        assert filled_static_values == expected_static_values
