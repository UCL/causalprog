"""Translating backend object syntax to frontend syntax."""

from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import signature
from typing import Any

from causalprog._abc.backend_agnostic import Backend, BackendAgnostic

from ._convert_signature import convert_signature
from ._typing import ParamNameMap, StaticValues


# TODO: Tests for this guy
@dataclass
class Translation:
    """
    Helper class for mapping frontend signatures to backend signatures.

    Predominantly a convenience wrapper for working with different backends.
    The attributes stored in an instance form the compulsory arguments that
    need to be passed to ``convert_signature`` in order to map a backend
    function to the frontend syntax.
    """

    backend_name: str
    frontend_name: str
    param_map: ParamNameMap
    frozen_args: StaticValues = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.backend_name = str(self.backend_name)
        self.frontend_name = str(self.frontend_name)
        self.frozen_args = dict(self.frozen_args)
        self.param_map = dict(self.param_map)

        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.param_map.items()
        ):
            msg = "Parameter map must map names to names (str -> str)"
            raise ValueError(msg)
        if not all(isinstance(key, str) for key in self.frozen_args):
            msg = "Frozen args must be specified by name (str)"
            raise ValueError(msg)


# TODO: tests for this guy after tests for the above guy!
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
        native: Backend,
        *translations: Translation,
    ) -> None:
        """
        Translate a backend object into a frontend-compatible object.

        Args:
            native (Backend): Backend object that must be translated to support frontend
                syntax.
            **translations (Translation): Keyword-specified ``Translation``s that map
                the methods of ``native`` to the (signatures of the) methods that the
                ``_frontend_provides``. Keyword names are interpreted as the name of the
                backend method to translate, whilst ``Translation.target_name`` is
                interpreted as the name of the frontend method that this backend method
                performs the role of.

        """
        super().__init__(backend=native)

        self.translations = {}
        self.frontend_to_native_names = {name: name for name in self._frontend_provides}
        for t in translations:
            native_name = t.backend_name
            translated_name = t.frontend_name
            native_method = getattr(self._backend_obj, native_name)
            target_signature = signature(getattr(self, translated_name))

            self.translations[translated_name] = convert_signature(
                native_method, target_signature, t.param_map, t.frozen_args
            )
            self.frontend_to_native_names[translated_name] = native_name

        # Methods without explicit translations are assumed to be the identity map
        for method in self._frontend_provides:
            if method not in self.translations:
                self.translations[method] = self.identity

        self.validate()

    def _call_backend_with(self, method: str, *args: Any, **kwargs: Any) -> Any:  # noqa:ANN401
        """Translate arguments and then call the backend."""
        backend_method = getattr(self._backend_obj, method)
        backend_args, backend_kwargs = self.translations[method](*args, **kwargs)
        return backend_method(*backend_args, **backend_kwargs)

    # IDEA NOW is that we could now define
    # def sample(*args, **kwargs):
    #    return self._call_backend_with("sample", *args, **kwargs)
