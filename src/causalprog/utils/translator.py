"""
Helper class to keep the codebase backend-agnostic.

Our frontend (or user-facing) classes each use a syntax that applies across the package
codebase. By contrast, the various backends that we want to support will have different
syntaxes and call signatures for the functions that we want to support. As such, we need
a helper class that can store this "translation" information, allowing the user to
interact with the package in a standard way but also allowing them to choose their own
backend if desired.
"""

import inspect
from abc import ABC, abstractmethod
from typing import Any


class Translator(ABC):
    """
    Maps syntax of a backend function to our frontend syntax.

    Different backends have different syntax for drawing samples from the distributions
    they support. In order to map these different syntaxes to our backend-agnostic
    framework, we need a container class to map the names we have chosen for our
    frontend methods to those used by their corresponding backend method.
    """

    backend_method: str
    corresponding_backend_arg: dict[str, str]

    @property
    @abstractmethod
    def _frontend_method(self) -> str:
        """Name of the frontend method that the backend is to be translated into."""
        return ""

    @property
    @abstractmethod
    def compulsory_frontend_args(self) -> set[str]:
        """Arguments that are required by the frontend function."""
        return set()

    @property
    def compulsory_backend_args(self) -> set[str]:
        """Arguments that are required to be taken by the backend function."""
        return {
            self.corresponding_backend_arg[arg_name]
            for arg_name in self.compulsory_frontend_args
        }

    def __init__(
        self, backend_method: str | None = None, **front_args_to_back_args: str
    ) -> None:
        """
        Create a new Translator.

        Args:
            backend_method (str): Name of the backend method that the instance
            translates.
            **front_args_to_back_args (str): Mapping of frontend argument names to the
            corresponding backend argument names.

        """
        # Assume backend name is identical to frontend name if not provided explicitly
        self.backend_method = (
            backend_method if backend_method else self._frontend_method
        )

        # This should really be immutable after we fill defaults!
        self.corresponding_backend_arg = dict(front_args_to_back_args)
        # Assume compulsory frontend args that are not given translations
        # retain their name in the backend.
        for arg in self.compulsory_frontend_args:
            if arg not in self.corresponding_backend_arg:
                self.corresponding_backend_arg[arg] = arg

    def translate_args(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Translate frontend arguments (with values) to backend arguments."""
        return {
            self.corresponding_backend_arg[arg_name]: arg_value
            for arg_name, arg_value in kwargs.items()
        }

    def validate_compatible(self, obj: object) -> None:
        """
        Determine if ``obj`` provides a compatible backend method.

        ``obj`` must provide a callable whose name matches ``self.backend_method``,
        and the callable referenced must take arguments matching the names specified in
        ``self.corresponding_backend_arg.values()``. Any arguments in
        ``self.backend_default_args`` must also be accepted by the callable.

        Args:
            obj (type): Object to check possesses a method that can be called with the
                information stored.

        """
        # Check that obj does provide a method of matching name
        if not hasattr(obj, self.backend_method):
            msg = f"{obj} has no method '{self.backend_method}'."
            raise AttributeError(msg)
        if not callable(getattr(obj, self.backend_method)):
            msg = f"'{self.backend_method}' attribute of {obj} is not callable."
            raise TypeError(msg)

        # Check that this method will be callable with the information given.
        method_params = inspect.signature(getattr(obj, self.backend_method)).parameters
        # The arguments that will be passed are actually taken by the method.
        for compulsory_arg in self.compulsory_backend_args:
            if compulsory_arg not in method_params:
                msg = (
                    f"'{self.backend_method}' does not "
                    f"take argument '{compulsory_arg}'."
                )
                raise TypeError(msg)
        # The method does not _require_ any additional arguments
        method_requires = {
            name for name, p in method_params.items() if p.default is p.empty
        }
        if not method_requires.issubset(self.compulsory_backend_args):
            args_not_accounted_for = method_requires - self.compulsory_backend_args
            raise TypeError(
                f"'{self.backend_method}' not provided compulsory arguments "
                "(missing " + ", ".join(args_not_accounted_for) + ")"
            )
