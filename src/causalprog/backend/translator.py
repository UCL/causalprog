"""Translating backend object syntax to frontend syntax."""

from collections.abc import Callable
from inspect import signature
from typing import Any

from causalprog._abc.backend_agnostic import Backend, BackendAgnostic

from ._convert_signature import convert_signature
from .translation import Translation


class Translator(BackendAgnostic[Backend]):
    """
    Translates the methods of a backend object into frontend syntax.

    A ``Translator`` acts as an intermediary between a backend that is supplied by the
    user and the frontend syntax that ``causalprog`` relies on. The default backend of
    ``causalprog`` uses a syntax compatible with ``jax``.

    Other backends may not conform to the syntax that ``causalprog`` expects, but
    nonetheless may provide the functionality that it requires. A ``Translator`` is able
    to make calls to (the relevant methods of) this backend, whilst still conforming to
    the frontend syntax of ``causalprog``.

    As an example, suppose that we have a frontend class ``C`` that needs to provide a
    method ``do_this``. ``causalprog`` expects ``C`` to provide the functionality
    of ``do_this`` via one of its methods, ``C.do_this(*c_args, **c_kwargs)``.
    Now suppose that a class ``D`` from a different, external package might also
    provides the functionality of ``do_this``, but it is done by calling
    ``D.do_this_different(*d_args, **d_kwargs)``, where there is some mapping
    ``m: *c_args, **c_kwargs -> *d_args, **d_kwargs``. In such a case, ``causalprog``
    needs to use a ``Translator`` ``T``, rather than ``D`` directly, where

    ``T.do_this(*c_args, **c_kwargs) = D.do_this_different(m(*c_args, **c_kwargs))``.
    """

    frontend_to_native_names: dict[str, str]
    translations: dict[str, Callable]

    @staticmethod
    def identity(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:  # noqa: ANN401
        """Identity map on positional and keyword arguments."""
        return args, kwargs

    def __init__(
        self,
        *translations: Translation,
        backend: Backend,
    ) -> None:
        """
        Translate a backend object into a frontend-compatible object.

        Args:
            backend (Backend): Backend object that must be translated to support
                frontend syntax.
            *translations (Translation): ``Translation``s that map the methods of
                ``backend`` to the (signatures of the) methods that the
                ``_frontend_provides``.

        """
        super().__init__(backend=backend)

        self.translations = {}
        self.frontend_to_native_names = {name: name for name in self._frontend_provides}
        for t in translations:
            native_name = t.backend_name
            translated_name = t.frontend_name
            native_method = getattr(self._backend_obj, native_name)
            target_method = getattr(self, translated_name)
            target_signature = signature(target_method)

            self.translations[translated_name] = convert_signature(
                native_method, target_signature, t.param_map, t.frozen_args
            )
            self.frontend_to_native_names[translated_name] = native_name

        # Methods without explicit translations are assumed to be the identity map,
        # provided they exist on the backend object.
        for method in self._frontend_provides:
            method_has_translation = method in self.translations
            backend_has_method = hasattr(self._backend_obj, method)
            if not (method_has_translation or backend_has_method):
                msg = (
                    f"No translation provided for {method}, "
                    "which the backend is lacking."
                )
                raise AttributeError(msg)
            if not method_has_translation:
                # Assume the identity mapping to the backend method, otherwise.
                self.translations[method] = self.identity

        self.validate()

    def _call_backend_with(self, method: str, *args: Any, **kwargs: Any) -> Any:  # noqa:ANN401
        """Translate arguments, then call the backend."""
        backend_method = getattr(
            self._backend_obj, self.frontend_to_native_names[method]
        )
        backend_args, backend_kwargs = self.translations[method](*args, **kwargs)
        return backend_method(*backend_args, **backend_kwargs)
