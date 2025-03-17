from abc import ABC, abstractmethod
from typing import Generic, TypeVar

Backend = TypeVar("Backend")


class BackendAgnostic(ABC, Generic[Backend]):
    """A frontend object that must be backend-agnostic."""

    _backend_obj: Backend

    def __init__(self, *, backend: Backend) -> None:
        self._backend_obj = backend

    @property
    @abstractmethod
    def _frontend_provides(self) -> tuple[str, ...]:
        """Methods that the frontend object must provide."""

    def _validate(self) -> bool:
        """Return ``True`` if all ``_frontend_provides`` methods are present."""
        return all(hasattr(self, attr) for attr in self._frontend_provides)

    def get_backend(self) -> Backend:
        """Access to the backend object."""
        return self._backend_obj
