"""C."""

from collections.abc import Callable

import jax
import numpy.typing as npt
from numpyro.infer import Predictive

from causalprog.causal_problem.causal_estimand import CausalEstimand, Constraint, Model


def sample_model(
    model: Predictive, rng_key: jax.Array, parameter_values: dict[str, npt.ArrayLike]
) -> dict[str, npt.ArrayLike]:
    return jax.vmap(lambda pv, key: model(key, **pv), in_axes=(None, 0))(
        parameter_values,
        jax.random.split(rng_key, model.num_samples),
    )


class CausalProblem:
    causal_estimand: CausalEstimand
    constraints: list[Constraint]

    def __init__(
        self,
        *constraints: Constraint,
        causal_estimand: CausalEstimand,
    ):
        self.causal_estimand = causal_estimand
        self.constraints = list(constraints)

    def lagrangian(
        self, n_samples: int = 1000
    ) -> Callable[[dict[str, npt.ArrayLike], Model, jax.Array], npt.ArrayLike]:
        """Assemble the Lagrangian."""

        def _inner(
            parameter_values: dict[str, npt.ArrayLike], model: Model, rng_key: jax.Array
        ) -> npt.ArrayLike:
            # In general, we will need to check which of our CE/CONs require masking,
            # and do multiple predictive models to account for this...
            # We can always pre-build the predictive models too, so we should replace
            # the "model" input with something that can map the right predictive models
            # to the CE/CONS that need them.
            predictive_model = Predictive(model=model, num_samples=n_samples)
            all_samples = sample_model(predictive_model, rng_key, parameter_values)

            value = self.causal_estimand.do_with_samples(**all_samples)
            # CLEANER IF THE LAGRANGE MULTIPLIERS COULD BE A SECOND FUNCTION ARG,
            # as right now they have to be inside the parameter dict...
            value += sum(
                parameter_values[f"_l_mult{i}"] * c.do_with_samples(**all_samples)
                for i, c in enumerate(self.constraints)
            )
            return value

        return _inner
