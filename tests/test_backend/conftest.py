from collections.abc import Callable
from inspect import Signature, signature

import pytest


@pytest.fixture
def general_function() -> Callable:
    def _general_function(
        posix, /, posix_def="posix_def", *vargs, kwo, kwo_def="kwo_def", **kwargs
    ):
        """Return the provided arguments."""
        return posix, posix_def, vargs, kwo, kwo_def, kwargs

    return _general_function


@pytest.fixture
def general_function_signature(general_function: Callable) -> Signature:
    """Signature of the ``general_function`` callable."""
    return signature(general_function)
