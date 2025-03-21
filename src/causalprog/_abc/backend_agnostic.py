from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

Backend = TypeVar("Backend")


class BackendAgnostic(ABC, Generic[Backend]):
    """
    A frontend object that must be backend-agnostic.

    ``BackendAgnostic`` is a means of ensuring that an object provides the functionality
    and interface that our package expects, irrespective of how this functionality is
    actually carried out. An instance of a ``BackendAgnostic`` class stores a reference
    to its ``_backend_obj``, and falls back on this object's methods and attributes if
    the instance itself does not possess the required attributes. Methods that the
    ``BackendAgnostic`` object can also be explicitly defined in the class, and make
    calls to the ``_backend_obj`` as necessary.
    """

    _backend_obj: Backend

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Fallback on the ``_backend_obj`` a frontend attribute isn't found."""
        if name in self._frontend_provides and hasattr(self._backend_obj, name):
            return getattr(self._backend_obj, name)
        msg = f"{self} has no attribute {name}."
        raise AttributeError(msg)

    def __init__(self, *, backend: Backend) -> None:
        self._backend_obj = backend

    @property
    @abstractmethod
    def _frontend_provides(self) -> tuple[str, ...]:
        """Names of attributes that an instance of this class must provide."""

    @property
    def _missing_attrs(self) -> set[str]:
        """Return the names of frontend attributes that are missing."""
        return {attr for attr in self._frontend_provides if not hasattr(self, attr)}

    def get_backend(self) -> Backend:
        """Access to the backend object."""
        return self._backend_obj

    def validate(self) -> None:
        """
        Determine if all expected frontend attributes are provided.

        Raises:
            AttributeError: If frontend methods are not present.

        """
        if len(self._missing_attrs) != 0:
            raise AttributeError(
                "Missing frontend methods: " + ", ".join(self._missing_attrs)
            )
