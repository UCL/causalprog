"""C."""

from collections.abc import Callable
from typing import Any, Concatenate, TypeAlias

import numpy.typing as npt

Model: TypeAlias = Callable[..., Any]
EffectHandler: TypeAlias = Callable[Concatenate[Model, ...], Model]
ModelMask: TypeAlias = tuple[EffectHandler, dict]


class _CPComponent:
    """"""

    do_with_samples: Callable[..., npt.ArrayLike]
    effect_handlers: tuple[ModelMask, ...]

    @property
    def requires_model_adaption(self) -> bool:
        """Return True if effect handlers need to be applied to model."""
        return len(self.effect_handlers) > 0

    def __init__(
        self,
        *effect_handlers: ModelMask,
        do_with_samples: Callable[..., npt.ArrayLike],
    ):
        self.effect_handlers = tuple(effect_handlers)
        self.do_with_samples = do_with_samples

    def apply_effects(self, model: Model) -> Model:
        """Apply any necessary effect handlers prior to evaluating."""
        adapted_model = model
        for handler, handler_options in self.effect_handlers:
            adapted_model = handler(adapted_model, **handler_options)
        return adapted_model


class CausalEstimand(_CPComponent):
    """"""


class Constraint(_CPComponent):
    """"""
