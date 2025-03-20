import re
from collections.abc import Iterable
from inspect import Parameter, Signature
from typing import Any

import pytest

from causalprog.backend._convert_signature import convert_signature


def general_function(
    posix, /, posix_def="posix_def", *vargs, kwo, kwo_def="kwo_def", **kwargs
):
    return posix, posix_def, vargs, kwo, kwo_def, kwargs


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
