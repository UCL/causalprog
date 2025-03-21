import re
from typing import Any

import pytest

from causalprog.backend.translation import Translation


@pytest.mark.parametrize(
    ("constructor_kwargs", "expected"),
    [
        pytest.param(
            {
                "backend_name": "backend",
                "frontend_name": "frontend",
                "param_map": {"0": "0", "1": "1"},
            },
            None,
            id="Respect default frozen args.",
        ),
        pytest.param(
            {
                "backend_name": "backend",
                "frontend_name": "frontend",
                "param_map": {},
                "frozen_args": {"0": 0, "1": 3.1415},
            },
            None,
            id="frozen_args dict-values can be Any.",
        ),
        pytest.param(
            {
                "backend_name": 100,
                "frontend_name": "frontend",
                "param_map": {"0": "0", "1": "1"},
            },
            TypeError("backend_name '100' is not a string."),
            id="Backend name must be string.",
        ),
        pytest.param(
            {
                "backend_name": "backend",
                "frontend_name": [1, 2, 3],
                "param_map": {"0": "0", "1": "1"},
            },
            TypeError("frontend_name '[1, 2, 3]' is not a string."),
            id="Frontend name must be string.",
        ),
        pytest.param(
            {
                "backend_name": "backend",
                "frontend_name": "frontend",
                "param_map": {"0": "0", "1": 1},
            },
            TypeError("Parameter map must map str -> str."),
            id="Parameter map value is not string.",
        ),
        pytest.param(
            {
                "backend_name": "backend",
                "frontend_name": "frontend",
                "param_map": {0: "0", "1": "1"},
            },
            TypeError("Parameter map must map str -> str."),
            id="Parameter map key is not string.",
        ),
        pytest.param(
            {
                "backend_name": "backend",
                "frontend_name": "frontend",
                "param_map": {},
                "frozen_args": {0: 0, "1": "1"},
            },
            TypeError("Frozen args must be specified by name (str)."),
            id="frozen_args dict-keys must be str.",
        ),
    ],
)
def test_translation(
    constructor_kwargs: dict[str, Any], expected: None | Exception
) -> None:
    if isinstance(expected, Exception):
        with pytest.raises(type(expected), match=re.escape(str(expected))):
            Translation(**constructor_kwargs)
    else:
        Translation(**constructor_kwargs)
