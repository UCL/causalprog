"""Classes for defining causal estimands and constraints of causal problems."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Concatenate, TypeAlias

import numpy.typing as npt

Model: TypeAlias = Callable[..., Any]
EffectHandler: TypeAlias = Callable[Concatenate[Model, ...], Model]


@dataclass
class HandlerToApply:
    """Specifies a handler than needs to be applied to a model at runtime."""

    handler: EffectHandler
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pair(cls, pair: tuple[EffectHandler, dict]) -> "HandlerToApply":
        """
        TODO: make pair just any lenght-2 container.

        and auto-identify which time is the options and which item is the callable
        """
        return cls(handler=pair[0], options=pair[1])

    def __post_init__(self) -> None:
        if not callable(self.handler):
            msg = f"{self.handler} is not callable!"
            raise TypeError(msg)
        if not isinstance(self.options, dict):
            msg = f"{self.options} should be keyword-argument mapping."
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
        return len(self._effect_handlers) > 0

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
        self._effect_handlers = tuple(
            h if isinstance(h, HandlerToApply) else HandlerToApply.from_pair(h)
            for h in effect_handlers
        )
        self._do_with_samples = do_with_samples

    def apply_effects(self, model: Model) -> Model:
        """Apply any necessary effect handlers prior to evaluating."""
        adapted_model = model
        for handler in self._effect_handlers:
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


class CausalEstimand(_CPComponent):
    """
    A Causal Estimand.

    The causal estimand is the function that we want to minimise (and maximise)
    as part of a causal problem. It should be a scalar-valued function of the
    random variables appearing in a graph.
    """


class Constraint(_CPComponent):
    r"""
    A Constraint that forms part of a causal problem.

    Constraints of a causal problem are derived properties of RVs for which we
    have observed data. The causal estimand is minimised (or maximised) subject
    to the predicted values of the constraints being close to their observed
    values in the data.

    Adding a constraint $g(\theta)$ to a causal problem (where $\theta$ are the
    parameters of the causal problem) essentially imposes an additional
    constraint on the minimisation problem;

    $$ g(\theta) - g_{\text{data}} \leq \epsilon, $$

    where $g_{\text{data}}$ is the observed data value for the quantity $g$,
    and $\epsilon$ is some tolerance.
    """

    # TODO: (https://github.com/UCL/causalprog/issues/89)
    # Should explain that Constraint needs more inputs and slightly different
    # interpretation of the `do_with_samples` object.
    # Inputs:
    # - include epsilon as an input (allows constraints to have different tolerances)
    # - `do_with_samples` should just be $g(\theta)$. Then have the instance build the
    #   full constraint that will need to be called in the Lagrangian.
    # - $g$ still needs to be scalar valued? Allow a wrapper function to be applied in
    #   the event $g$ is vector-valued.
    # If we do this, will also need to override __call__...
