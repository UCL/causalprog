import re
from collections.abc import Iterable
from inspect import Parameter, Signature, signature
from typing import Any

import pytest

from causalprog.backend._convert_signature import (
    _check_variable_length_params,
    _signature_can_be_cast,
    convert_signature,
)
from causalprog.backend._typing import ParamNameMap, StaticValues


def general_function(
    posix, /, posix_def="posix_def", *vargs, kwo, kwo_def="kwo_def", **kwargs
):
    return posix, posix_def, vargs, kwo, kwo_def, kwargs


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
            signature(general_function),
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
def test_validate_variable_length_parameters(
    signature: Signature, expected: Exception | dict
):
    if isinstance(expected, Exception):
        with pytest.raises(type(expected), match=re.escape(str(expected))):
            _check_variable_length_params(signature)
    else:
        returned_names = _check_variable_length_params(signature)

        assert returned_names == expected


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
            signature(general_function),
            signature(general_function),
            {},
            {},
            ({key: key for key in signature(general_function).parameters}, {}),
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
        computed_output = _signature_can_be_cast(
            signature_to_convert,
            new_signature,
            param_name_map,
            give_static_value,
        )

        assert computed_output == expected_output


_kwargs_static_value = {"some": "keyword-arguments"}


@pytest.mark.parametrize(
    (
        "posix_for_new_call",
        "keyword_for_new_call",
        "expected_assignments",
    ),
    [
        pytest.param(
            [1, 2],
            {"kwo_n": 3, "kwo_def_n": 4},
            {
                "posix": 3,
                "posix_def": 4,
                "vargs": (),
                "kwo": 1,
                "kwo_def": 2,
                "kwargs": _kwargs_static_value,
            },
            id="No vargs supplied.",
        ),
        pytest.param(
            [1, 2, 10, 11, 12],
            {"kwo_n": 3, "kwo_def_n": 4},
            {
                "posix": 3,
                "posix_def": 4,
                "vargs": (10, 11, 12),
                "kwo": 1,
                "kwo_def": 2,
                "kwargs": _kwargs_static_value,
            },
            id="Supply vargs.",
        ),
        pytest.param(
            [1],
            {"kwo_n": 3},
            {
                "posix": 3,
                "posix_def": "default_for_kwo_def_n",
                "vargs": (),
                "kwo": 1,
                "kwo_def": "default_for_posix_def_n",
                "kwargs": _kwargs_static_value,
            },
            id="New default values respected.",
        ),
        pytest.param(
            [1],
            {"kwo_n": 3, "extra_kwarg": "not allowed"},
            TypeError("got an unexpected keyword argument 'extra_kwarg'"),
            id="kwargs not allowed in new signature.",
        ),
        pytest.param(
            [1, 2],
            {"kwo_n": 3, "posix_def_n": 2},
            TypeError("multiple values for argument 'posix_def_n'"),
            id="Multiple values for new parameter.",
        ),
    ],
)
def test_convert_signature(
    posix_for_new_call: Iterable[Any],
    keyword_for_new_call: dict[str, Any],
    expected_assignments: dict[str, Any] | Exception,
) -> None:
    """
    To ease the burden of setting up and parametrising this test,
    we will always use the general_function signature as the target and source
    signature.

    However, the target signature will swap the roles of the positional and keyword
    parameters, essentially mapping:

    ``posix, posix_def, *vargs, kwo, kwo_def, **kwargs``

    to

    ``kwo_n, kwo_def_n, *vargs_n, posix_n, posix_def_n``.

    ``give_static_value`` will give kwargs a default value.

    We can then make calls to this new signature, and since ``general_function`` returns
    the arguments it received, we can validate that correct passing of arguments occurs.
    """
    param_name_map = {
        "posix": "kwo_n",
        "posix_def": "kwo_def_n",
        "kwo": "posix_n",
        "kwo_def": "posix_def_n",
    }
    give_static_value = {"kwargs": _kwargs_static_value}
    new_signature = Signature(
        [
            Parameter("posix_n", Parameter.POSITIONAL_ONLY),
            Parameter(
                "posix_def_n",
                Parameter.POSITIONAL_OR_KEYWORD,
                default="default_for_posix_def_n",
            ),
            Parameter("vargs_n", Parameter.VAR_POSITIONAL),
            Parameter("kwo_n", Parameter.KEYWORD_ONLY),
            Parameter(
                "kwo_def_n", Parameter.KEYWORD_ONLY, default="default_for_kwo_def_n"
            ),
        ]
    )
    new_function = convert_signature(
        general_function, new_signature, param_name_map, give_static_value
    )

    if isinstance(expected_assignments, Exception):
        with pytest.raises(
            type(expected_assignments), match=re.escape(str(expected_assignments))
        ):
            new_function(*posix_for_new_call, **keyword_for_new_call)
    else:
        posix, posix_def, vargs, kwo, kwo_def, kwargs = new_function(
            *posix_for_new_call, **keyword_for_new_call
        )
        assert posix == expected_assignments["posix"]
        assert posix_def == expected_assignments["posix_def"]
        assert vargs == expected_assignments["vargs"]
        assert kwo == expected_assignments["kwo"]
        assert kwo_def == expected_assignments["kwo_def"]
        assert kwargs == expected_assignments["kwargs"]
