"""Tests for the SampleCompatibility class."""

import re

import pytest

from causalprog.distributions.base import SampleCompatibility


class DummyClass:
    """
    Stub class for testing.

    Intended use is to provide a variety of method call signatures, that can be used to
    verify whether the ``SampleCompatibility`` class is correctly able to determine if
    it will be able to call an underlying method without error.
    """

    @property
    def prop(self) -> None:
        """Properties are not callable."""
        return

    def __init__(self) -> None:
        """Create an instance."""
        return

    def __str__(self) -> str:
        """Display, in case the object appears in an error string."""
        return "DummyClass instance"

    def method_two_args(self, arg1: int, arg2: int) -> int:
        """Take 2 compulsory arguments."""
        return arg1 + arg2

    def method_two_args_one_compulsory(self, arg1: int, arg2: int = 0) -> int:
        """Take 1 compulsory and 1 optional argument."""
        return arg1 + arg2

    def method_args_and_kwargs(self, arg1: int, arg2: int = 0, kwarg1: int = 0) -> int:
        """Take 1 compulsory and 2 optional arguments."""
        return arg1 + arg2 + kwarg1


INSTANCE = DummyClass()


@pytest.mark.parametrize(
    ("info", "obj", "expected_result"),
    [
        pytest.param(
            SampleCompatibility(method="method_does_not_exist"),
            INSTANCE,
            AttributeError(
                "Distribution-defining object DummyClass instance "
                "has no method 'method_does_not_exist'."
            ),
            id="No appropriate method",
        ),
        pytest.param(
            SampleCompatibility(method="prop"),
            INSTANCE,
            TypeError("'prop' attribute of DummyClass instance is not callable."),
            id="Method not callable",
        ),
        pytest.param(
            SampleCompatibility(
                method="method_two_args", rng_key="arg1", sample_shape="not_an_arg"
            ),
            INSTANCE,
            TypeError("'method_two_args' does not take argument 'not_an_arg'"),
            id="Does not take compulsory argument",
        ),
        pytest.param(
            SampleCompatibility(
                method="method_two_args", rng_key="arg1", sample_shape="not_an_arg"
            ),
            INSTANCE,
            TypeError("'method_two_args' does not take argument 'not_an_arg'"),
            id="Missing compulsory argument",
        ),
        pytest.param(
            SampleCompatibility(
                method="method_args_and_kwargs",
                rng_key="arg2",
                sample_shape="kwarg1",
            ),
            INSTANCE,
            TypeError(
                "'method_args_and_kwargs' not provided compulsory arguments "
                "(missing arg1)"
            ),
            id="Must supply all compulsory arguments",
        ),
        pytest.param(
            SampleCompatibility(
                method="method_two_args", rng_key="arg1", sample_shape="arg2"
            ),
            INSTANCE,
            None,
            id="Provide both compulsory arguments",
        ),
        pytest.param(
            SampleCompatibility(
                method="method_two_args_one_compulsory",
                rng_key="arg1",
                sample_shape="arg2",
            ),
            INSTANCE,
            None,
            id="Provide all compulsory arguments, and one optional argument",
        ),
        pytest.param(
            SampleCompatibility(
                method="method_args_and_kwargs",
                rng_key="arg1",
                sample_shape="kwarg1",
            ),
            INSTANCE,
            None,
            id="Provide all compulsory arguments, and one keyword argument",
        ),
    ],
)
def test_validate_compatible(
    info: SampleCompatibility, obj: object, expected_result: Exception | None
) -> None:
    """
    Test the validate_compatible method.

    Test that a SampleCompatibility instance correctly determines if a given method
    of a given object is callable, with the information stored in the instance.
    """
    if expected_result is not None:
        with pytest.raises(
            type(expected_result), match=re.escape(str(expected_result))
        ):
            info.validate_compatible(obj)
    else:
        info.validate_compatible(obj)
