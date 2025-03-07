"""Tests for the SampleCompatibility class."""

import re

import pytest

from causalprog.utils.translator import Translator


class _TranslatorForTesting(Translator):
    @property
    def _frontend_method(self) -> str:
        """Name of the frontend method that the backend is to be translated into."""
        return "method"

    @property
    def compulsory_frontend_args(self) -> set[str]:
        """Arguments that are required by the frontend function."""
        return {"arg1", "arg2"}


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

    def method(self, arg1: int, arg2: int = 0, kwarg1: int = 0) -> int:
        """Take 1 compulsory and 2 optional arguments."""
        return arg1 + arg2 + kwarg1


@pytest.fixture
def dummy_class_instance() -> DummyClass:
    """Instance of the ``DummyClass`` to use in testing."""
    return DummyClass()


@pytest.mark.parametrize(
    ("info", "expected_result"),
    [
        pytest.param(
            _TranslatorForTesting(backend_method="method_does_not_exist"),
            AttributeError(
                "Distribution-defining object DummyClass instance "
                "has no method 'method_does_not_exist'."
            ),
            id="Object does not have the backend method.",
        ),
        pytest.param(
            _TranslatorForTesting(backend_method="prop"),
            TypeError("'prop' attribute of DummyClass instance is not callable."),
            id="Object backend method is not callable.",
        ),
        pytest.param(
            _TranslatorForTesting(arg1="not_an_arg"),
            TypeError("'method' does not take argument 'not_an_arg'"),
            id="Backend does not take compulsory argument.",
        ),
        pytest.param(
            _TranslatorForTesting("method", arg1="arg2", arg2="kwarg1"),
            TypeError("'method' not provided compulsory arguments (missing arg1)"),
            id="Backend cannot have unspecified compulsory arguments.",
        ),
        pytest.param(
            _TranslatorForTesting(),
            None,
            id="Fall back on defaults.",
        ),
        pytest.param(
            _TranslatorForTesting(backend_method="method", arg2="kwarg1"),
            None,
            id="Match args out-of-order.",
        ),
    ],
)
def test_validate_compatible(
    info: _TranslatorForTesting,
    dummy_class_instance: DummyClass,
    expected_result: Exception | None,
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
            info.validate_compatible(dummy_class_instance)
    else:
        info.validate_compatible(dummy_class_instance)
