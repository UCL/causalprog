"""C."""

from collections.abc import Callable

import jax
import numpy.typing as npt
from numpyro.infer import Predictive

from causalprog.causal_problem.causal_estimand import CausalEstimand, Constraint, Model


def sample_model(
    model: Predictive, rng_key: jax.Array, parameter_values: dict[str, npt.ArrayLike]
) -> dict[str, npt.ArrayLike]:
    """
    Draw samples from the predictive model.

    TODO: Move function to somewhere more appropriate.

    Args:
        model: Predictive model to draw samples from.
        rng_key: PRNG Key to use in pseudorandom number generation.
        parameter_values: Model parameter values to substitute.

    Returns:
        `dict` of samples, with RV labels as keys and sample values (`jax.Array`s) as
            values.

    """
    return jax.vmap(lambda pv, key: model(key, **pv), in_axes=(None, 0))(
        parameter_values,
        jax.random.split(rng_key, model.num_samples),
    )


class CausalProblem:
    """Defines a causal problem."""

    causal_estimand: CausalEstimand
    constraints: list[Constraint]

    def __init__(
        self,
        *constraints: Constraint,
        causal_estimand: CausalEstimand,
    ) -> None:
        """Create a new causal problem."""
        self.causal_estimand = causal_estimand
        self.constraints = list(constraints)

    def lagrangian(
        self, n_samples: int = 1000, *, maximum_problem: bool = False
    ) -> Callable[
        [dict[str, npt.ArrayLike], npt.ArrayLike, Model, jax.Array], npt.ArrayLike
    ]:
        """
        Return a function that evaluates the Lagrangian of this `CausalProblem`.

        Following the
        [KKT theorem](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions),
        given the causal estimand and the constraints we can assemble a Lagrangian and
        seek its stationary points, to in turn identify minimisers of the constrained
        optimisation problem that we started with.

        The Lagrangian returned is a mathematical function of its first two arguments.
        The first argument is the same dictionary of parameters that is passed to models
        like `Graph.model`, and is the values the parameters (represented by the
        `ParameterNode`s) are taking. The second argument is a 1D vector of Lagrange
        multipliers, whose length is equal to the number of constraints.

        The remaining two arguments of the Lagrangian are the underlying model that it
        should use to generate and draw samples of the RVs from, and the PRNGKey that
        should be used in this generation.

        Note that our current implementation assumes there are no equality constraints
        being imposed (in which case, we would need a 3-argument Lagrangian function).

        TODO: Can we store g.model in the class, and have this still work???
        """
        maximisation_prefactor = -1.0 if maximum_problem else 1.0

        def _inner(
            parameter_values: dict[str, npt.ArrayLike],
            l_mult: jax.Array,
            model: Model,
            rng_key: jax.Array,
        ) -> npt.ArrayLike:
            # In general, we will need to check which of our CE/CONs require masking,
            # and do multiple predictive models to account for this...
            # We can always pre-build the predictive models too, so we should replace
            # the "model" input with something that can map the right predictive models
            # to the CE/CONS that need them.
            predictive_model = Predictive(model=model, num_samples=n_samples)
            all_samples = sample_model(predictive_model, rng_key, parameter_values)

            value = maximisation_prefactor * self.causal_estimand.do_with_samples(
                **all_samples
            )
            # CLEANER IF THE LAGRANGE MULTIPLIERS COULD BE A SECOND FUNCTION ARG,
            # as right now they have to be inside the parameter dict...
            # Cleaner if we could somehow build a vector-valued function of the
            # constraints and then take a dot product, but this works for now
            value += sum(
                l_mult[i] * c.do_with_samples(**all_samples)
                for i, c in enumerate(self.constraints)
            )
            return value

        return _inner
