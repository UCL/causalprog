"""Classes for defining causal estimands and constraints of causal problems."""

from collections.abc import Callable
from typing import Any, Concatenate, TypeAlias

import jax.numpy as jnp
import numpy.typing as npt

from causalprog.causal_problem._base_component import _CPComponent

Model: TypeAlias = Callable[..., Any]
EffectHandler: TypeAlias = Callable[Concatenate[Model, ...], Model]
ModelMask: TypeAlias = tuple[EffectHandler, dict]


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

    data: npt.ArrayLike
    tolerance: npt.ArrayLike
    _outer_norm: Callable[[npt.ArrayLike], float]

    def __init__(
        self,
        *effect_handlers: ModelMask,
        model_quantity: Callable[..., npt.ArrayLike],
        outer_norm: Callable[[npt.ArrayLike], float] | None = None,
        data: npt.ArrayLike = 0.0,
        tolerance: float = 1.0e-6,
    ) -> None:
        r"""
        Create a new constraint.

        Constraints have the form

        $$ c(\theta) :=
        \mathrm{norm}\left( g(\theta)
        - g_{\mathrm{data}} \right)
        - \epsilon $$

        where;
        - $\mathrm{norm}$ is the outer norm of the constraint (`outer_norm`),
        - $g(\theta)$ is the model quantity involved in the constraint
            (`model_quantity`),
        - $g_{\mathrm{data}}$ is the observed data (`data`),
        - $\epsilon$ is the tolerance in the data (`tolerance`).

        In a causal problem, each constraint appears as the condition $c(\theta)\leq 0$
        in the minimisation / maximisation (hence the inclusion of the $-\epsilon$
        term within $c(\theta)$ itself).

        $g$ should be a (possibly vector-valued) function that acts on (a subset of)
        samples from the random variables of the causal problem. It must accept
        variable keyword-arguments only, and should access the samples for each random
        variable by indexing via the RV names (node labels). It should return the
        model quantity as computed from the samples, that $g_{\mathrm{data}}$ observed.

        $g_{\mathrm{data}}$ should be a fixed value whose shape is broadcast-able with
        the return shape of $g$. It defaults to $0$ if not explicitly set.

        $\mathrm{norm}$ should be a suitable norm to take on the difference between the
        model quantity as predicted by the samples ($g$) and the observed data
        ($g_{\mathrm{data}}$). It must return a scalar value. The default is the 2-norm.
        """
        super().__init__(*effect_handlers, do_with_samples=model_quantity)

        if outer_norm is not None:
            self._outer_norm = outer_norm
        else:
            self._outer_norm = jnp.linalg.vector_norm

        self.data = data
        self.tolerance = tolerance

    def __call__(self, samples: dict[str, npt.ArrayLike]) -> npt.ArrayLike:
        """
        Evaluate the constraint, given RV samples.

        Args:
            samples: Mapping of RV (node) labels to drawn samples.

        Returns:
            Value of the constraint.

        """
        return (
            self._outer_norm(self._do_with_samples(**samples) - self.data)
            - self.tolerance
        )
