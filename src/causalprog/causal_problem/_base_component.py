"""Base class for components of causal problems."""

from collections.abc import Callable
from typing import Any

import numpy.typing as npt

from causalprog.causal_problem.handlers import EffectHandler, HandlerToApply, Model


class _CPComponent:
    """
    Base class for components of a Causal Problem.

    A _CPComponent has an attached method that it can apply to samples
    (`do_with_samples`), which will be passed sample values of the RVs
    during solution of a Causal Problem and used to evaluate the causal
    estimand or constraint the instance represents.

    It also has a sequence of effect handlers that need to be applied
    to the sampling model before samples can be drawn to evaluate this
    component. For example, if a component requires conditioning on the
    value of a RV, the `condition` handler needs to be applied to the
    underlying model, before generating samples to pass to the
    `do_with_sample` method. `effect_handlers` will be applied to the model
    in the order they are given.
    """

    do_with_samples: Callable[..., npt.ArrayLike]
    effect_handlers: tuple[HandlerToApply, ...]

    @property
    def requires_model_adaption(self) -> bool:
        """Return True if effect handlers need to be applied to model."""
        return len(self.effect_handlers) > 0

    def __call__(self, samples: dict[str, npt.ArrayLike]) -> npt.ArrayLike:
        """
        Evaluate the estimand or constraint, given sample values.

        Args:
            samples: Mapping of RV (node) labels to samples of that RV.

        Returns:
            Value of the estimand or constraint, given the samples.

        """
        return self._do_with_samples(**samples)

    def __init__(
        self,
        *effect_handlers: HandlerToApply | tuple[EffectHandler, dict[str, Any]],
        do_with_samples: Callable[..., npt.ArrayLike],
    ) -> None:
        self.effect_handlers = tuple(
            h if isinstance(h, HandlerToApply) else HandlerToApply.from_pair(h)
            for h in effect_handlers
        )
        self._do_with_samples = do_with_samples

    def apply_effects(self, model: Model) -> Model:
        """Apply any necessary effect handlers prior to evaluating."""
        adapted_model = model
        for handler in self.effect_handlers:
            adapted_model = handler.handler(adapted_model, handler.options)
        return adapted_model

    def can_use_same_model_as(self, other: "_CPComponent") -> bool:
        """
        Determine if two components use the same (predictive) model.

        Two components rely on the same model if they apply the same handlers
        to the model, which occurs if and only if `self.effect_handlers` and
        `other.effect_handlers` contain identical entries, in the same order.
        """
        if (not isinstance(other, _CPComponent)) or (
            len(self.effect_handlers) != len(other.effect_handlers)
        ):
            return False

        return all(
            my_handler == their_handler
            for my_handler, their_handler in zip(
                self.effect_handlers, other.effect_handlers, strict=True
            )
        )
