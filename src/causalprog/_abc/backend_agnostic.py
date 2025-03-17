from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

Backend = TypeVar("Backend")


class BackendAgnostic(ABC, Generic[Backend]):
    """A frontend object that must be backend-agnostic."""

    __slots__ = ("_backend_obj",)
    _backend_obj: Backend

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Fallback on the backend object a frontend method isn't found."""
        if name in self._frontend_provides and hasattr(self._backend_obj, name):
            return getattr(self._backend_obj, name)
        msg = f"{self} has no attribute {name}."
        raise AttributeError(msg)

    def __init__(self, *, backend: Backend) -> None:
        self._backend_obj = backend

    @property
    @abstractmethod
    def _frontend_provides(self) -> tuple[str, ...]:
        """Methods that an instance of this class must provide."""

    @property
    def _missing_methods(self) -> set[str]:
        """Return the names of frontend methods that are missing."""
        return {attr for attr in self._frontend_provides if not hasattr(self, attr)}

    def get_backend(self) -> Backend:
        """Access to the backend object."""
        return self._backend_obj

    def validate(self) -> None:
        """
        Determine if all expected frontend methods are provided.

        Raises:
            AttributeError: If frontend methods are not present.

        """
        if len(self._missing_methods) != 0:
            raise AttributeError(
                "Missing frontend methods: " + ", ".join(self._missing_methods)
            )
