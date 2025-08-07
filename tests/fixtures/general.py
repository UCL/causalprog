import re
from collections.abc import Callable
from contextlib import AbstractContextManager

import jax.random as jrn
import pytest


@pytest.fixture
def seed() -> int:
    return 0


@pytest.fixture
def rng_key(seed: int):
    return jrn.key(seed)


@pytest.fixture(scope="session")
def raises_context() -> Callable[[Exception], AbstractContextManager]:
    """Wrapper around `pytest.raises` that can be passed an Exception instance.

    For a variable containing an `Exception` instance,

    >>> expected_exception = Exception("message")

    the following statements are equivalent:

    >>> pytest.raises(
    ...     type(expected_exception),
    ...     match=re.escape(str(expected_exception))
    ... )
    >>> raises_context(expected_exception)
    """

    def _inner(match_error: Exception) -> AbstractContextManager:
        return pytest.raises(type(match_error), match=re.escape(str(match_error)))

    return _inner
