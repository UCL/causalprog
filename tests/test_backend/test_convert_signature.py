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
            Signature(),
            Signature(
                [
                    Parameter("vargs1", Parameter.VAR_POSITIONAL),
                    Parameter("vargs2", Parameter.VAR_POSITIONAL),
                ],
            ),
            {},
            {},
            ValueError("New signature takes more than 1 VAR_POSITIONAL argument."),
            id="Signature takes more than 1 VAR argument.",
        ),
        pytest.param(
            Signature(),
            Signature(),
            {},
            {},
            ValueError(
                "Either both signatures, or neither, must accept a "
                "variable number of positional arguments."
            ),
            id="Varg mismatch [new doesn't take].",
        ),
        pytest.param(
            Signature(),
            Signature(),
            {},
            {},
            ValueError(
                "Variable-positional parameter ({old_varg_param}) is not mapped "
                "to another variable-positional parameter."
            ),
            id="Varg mismatch [names are not mapped to each other].",
        ),
        pytest.param(
            Signature(),
            Signature(),
            {},
            {},
            ValueError("{p_name} is not mapped to a parameter in the new signature!"),
            id="Parameter not matched [positional only]",
        ),
        pytest.param(
            Signature(),
            Signature(),
            {},
            {},
            ValueError("{p_name} is not mapped to a parameter in the new signature!"),
            id="Parameter not matched [positional or keyword]",
        ),
        pytest.param(
            Signature(),
            Signature(),
            {},
            {},
            ValueError("{p_name} is not mapped to a parameter in the new signature!"),
            id="Parameter not matched [keyword only]",
        ),
        pytest.param(
            signature(general_function),
            signature(general_function),
            {},
            {},
            ("vargs", {key: key for key in signature(general_function).parameters}, {}),
            id="Can map to yourself.",
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
