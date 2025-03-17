from abc import ABC


class BackendAgnostic(ABC):
    """A frontend object that must be backend-agnostic."""

    _backend_obj: object

    def __init__(self, *, backend: object) -> None:
        self._backend_obj = backend
