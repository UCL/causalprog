"""Translating backend object syntax to frontend syntax."""

from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import Signature
from typing import Any

from causalprog._abc.backend_agnostic import Backend, BackendAgnostic

from ._typing import ParamNameMap, ReturnType, StaticValues
from .signature import convert_signature


@dataclass
class Translation:
    """
    Helper class for mapping frontend signatures to backend signatures.

    Predominantly a convenience wrapper for working with different backends.
    The attributes stored in an instance form the compulsory arguments that
    need to be passed to ``convert_signature`` in order to map a backend
    function to the frontend syntax.
    """

    target_signature: Signature
    param_map: ParamNameMap
    frozen_args: StaticValues = field(default_factory=dict)
    target_name: str | None = None

    def __post_init__(self) -> None:
        self.param_map = dict(self.param_map)
        self.frozen_args = dict(self.frozen_args)
        self.target_name = str(self.target_name) if self.target_name else None

        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.param_map.items()
        ):
            msg = "Parameter map must map names to names (str -> str)"
            raise ValueError(msg)
        if not all(isinstance(key, str) for key in self.frozen_args):
            msg = "Frozen args must be specified by name (str)"
            raise ValueError(msg)

    def translate(self, fn: Callable[..., ReturnType]) -> Callable[..., ReturnType]:
        """Convert a (compatible) callable's signature into the target_signature."""
        return convert_signature(
            fn, self.target_signature, self.param_map, self.frozen_args
        )


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

    translations: dict[str, Callable]

    def __init__(
        self,
        native: Backend,
        **known_translations: Translation,
    ) -> None:
        """
        Translate a backend object into a frontend-compatible object.

        Args:
            native (Backend): Backend object that must be translated to support frontend
                syntax.
            **known_translations (Translation): Keyword-specified ``Translation``
                objects that map the methods of ``native`` to the (signatures of the)
                methods that the ``_frontend_provides``. Keyword names are interpreted
                as the name of the backend method to translate, whilst
                ``Translation.target_name`` is interpreted as the name of the frontend
                method that this backend method performs the role of.

        """
        super().__init__(backend=native)

        self.translations = {}
        for native_name, t in known_translations.items():
            translated_name = t.target_name if t.target_name else native_name
            native_method = getattr(self._backend_obj, native_name)

            if translated_name in self.translations:
                msg = f"Method {translated_name} provided twice."
                raise ValueError(msg)
            self.translations[translated_name] = convert_signature(
                native_method, t.target_signature, t.param_map, t.frozen_args
            )

        self.validate()

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        # Check for translations before falling back on backend directly.
        if name in self.translations:
            return self.translations[name]
        return super().__getattr__(name)
