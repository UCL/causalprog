"""Translating backend object syntax to frontend syntax."""

from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import Signature

from ._typing import ParamNameMap, ReturnType, StaticValues
from .signature import convert_signature


@dataclass
class Translation:
    """Helper class for mapping frontend signatures to backend signatures."""

    target_signature: Signature
    param_map: ParamNameMap
    frozen_args: StaticValues = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.param_map = dict(self.param_map)
        self.frozen_args = dict(self.frozen_args)

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


class Translator:
    """Translates the methods of a backend object into frontend syntax."""

    known_translations: dict[str, Translation]
    target_class: object

    def translate(self, backend_obj: object):
        pass
