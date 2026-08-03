from inspect import Parameter, Signature

import pytest

from causalprog.backend._convert_signature import _check_variable_length_params


@pytest.mark.parametrize(
    ("signature", "expected"),
    [
        pytest.param(
            "general_function_signature",
            {Parameter.VAR_POSITIONAL: "vargs", Parameter.VAR_KEYWORD: "kwargs"},
            id="Valid, but complex, signature.",
        ),
    ],
)
def test_check_variable_length_parameters(
    signature: Signature,
    expected: Exception | dict,
    request,
    raises_context,
):
    if isinstance(signature, str):
        signature = request.getfixturevalue(signature)

    if isinstance(expected, Exception):
        with raises_context(expected):
            _check_variable_length_params(signature)
    else:
        returned_names = _check_variable_length_params(signature)

        assert returned_names == expected
