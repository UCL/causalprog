from inspect import Parameter, Signature

import pytest

from causalprog.backend._convert_signature import _check_variable_length_params


@pytest.mark.parametrize(
    ("signature", "expected"),
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
            "general_function_signature",
            {Parameter.VAR_POSITIONAL: "vargs", Parameter.VAR_KEYWORD: "kwargs"},
            id="Valid, but complex, signature.",
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
