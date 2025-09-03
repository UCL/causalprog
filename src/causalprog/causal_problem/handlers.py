"""Container class for specifying effect handlers that need to be applied at runtime."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Concatenate, TypeAlias

Model: TypeAlias = Callable[..., None]
EffectHandler: TypeAlias = Callable[Concatenate[Model, ...], Model]


@dataclass
class HandlerToApply:
    """Specifies a handler that needs to be applied to a model at runtime."""

    handler: EffectHandler
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pair(cls, pair: Sequence) -> "HandlerToApply":
        """
        Create an instance from an effect handler and its options.

        The two objects should be passed in as the elements of a container of length
        2. They can be passed in any order;
        - One element must be a dictionary, which will be interpreted as the `options`
            for the effect handler.
        - The other element must be callable, and will be interpreted as the `handler`
            itself.

        Args:
            pair: Container of two elements, one being the effect handler callable and
                the other being the options to pass to it (as a dictionary).

        Returns:
            Class instance corresponding to the effect handler and options passed.

        """
        if len(pair) != 2:  # noqa: PLR2004
            msg = (
                f"{cls.__name__} can only be constructed from a container of 2 elements"
            )
            raise ValueError(msg)

        # __post_init__ will catch cases when the incorrect types for one or both items
        # is passed, so we can just naively if-else here.
        handler: EffectHandler
        options: dict
        if callable(pair[0]):
            handler = pair[0]
            options = pair[1]
        else:
            handler = pair[1]
            options = pair[0]

        return cls(handler=handler, options=options)

    def __post_init__(self) -> None:
        """
        Validate set attributes.

        - The handler is a callable object.
        - The options have been passed as a dictionary of keyword-value pairs.
        """
        if not callable(self.handler):
            msg = f"{type(self.handler).__name__} is not callable."
            raise TypeError(msg)
        if not isinstance(self.options, dict):
            msg = (
                "Options should be dictionary mapping option arguments to values "
                f"(got {type(self.options).__name__})."
            )
            raise TypeError(msg)

    def __eq__(self, other: object) -> bool:
        """
        Equality operation.

        `HandlerToApply`s are considered equal if they use the same handler function and
        provide the same options to this function.

        Comparison to other types returns `False`.
        """
        return (
            isinstance(other, HandlerToApply)
            and self.handler is other.handler
            and self.options == other.options
        )
