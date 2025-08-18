"""C."""

from collections.abc import Callable
from typing import Any, Concatenate, TypeAlias

import numpy.typing as npt

Model: TypeAlias = Callable[..., Any]
EffectHandler: TypeAlias = Callable[Concatenate[Model, ...], Model]
ModelMask: TypeAlias = tuple[EffectHandler, dict]


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

    _do_with_samples: Callable[..., npt.ArrayLike]
    _effect_handlers: tuple[ModelMask, ...]

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
        *effect_handlers: ModelMask,
        do_with_samples: Callable[..., npt.ArrayLike],
    ) -> None:
        self._effect_handlers = tuple(effect_handlers)
        self._do_with_samples = do_with_samples

    def apply_effects(self, model: Model) -> Model:
        """Apply any necessary effect handlers prior to evaluating."""
        adapted_model = model
        for handler, handler_options in self._effect_handlers:
            adapted_model = handler(adapted_model, **handler_options)
        return adapted_model


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

    # TODO: Should explain that Constraint needs more inputs and slightly different
    # interpretation of the `do_with_samples` object.
    # Inputs:
    # - include epsilon as an input (allows constraints to have different tolerances)
    # - `do_with_samples` should just be $g(\theta)$. Then have the instance build the
    #   full constraint that will need to be called in the Lagrangian.
    # - $g$ still needs to be scalar valued? Allow a wrapper function to be applied in
    #   the event $g$ is vector-valued.
    # If we do this, will also need to override __call__...
