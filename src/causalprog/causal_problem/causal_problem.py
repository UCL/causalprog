"""Classes for representing causal problems."""

from collections.abc import Callable

import jax
import numpy.typing as npt
from numpyro.infer import Predictive

# TODO: Rename module to components
from causalprog.causal_problem.causal_estimand import (
    CausalEstimand,
    Constraint,
    _CPComponent,
)
from causalprog.graph import Graph


# TODO: https://github.com/UCL/causalprog/issues/88
def sample_model(
    model: Predictive, rng_key: jax.Array, parameter_values: dict[str, npt.ArrayLike]
) -> dict[str, npt.ArrayLike]:
    """
    Draw samples from the predictive model.

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

    _underlying_graph: Graph
    causal_estimand: CausalEstimand
    constraints: list[Constraint]

    @property
    def _ordered_components(self) -> list[_CPComponent]:
        """Internal ordering for components of the causal problem."""
        return [*self.constraints, self.causal_estimand]

    def __init__(
        self,
        graph: Graph,
        *constraints: Constraint,
        causal_estimand: CausalEstimand,
    ) -> None:
        """Create a new causal problem."""
        self._underlying_graph = graph
        self.causal_estimand = causal_estimand
        self.constraints = list(constraints)

    def _associate_models_to_components(
        self, n_samples: int
    ) -> tuple[list[Predictive], list[int]]:
        """
        Create models to be used by components of the problem.

        Depending on how many constraints (and the causal estimand) require effect
        handlers to wrap `self._underlying_graph.model`, we will need to create several
        predictive models to sample from. However, we also want to minimise the number
        of such models we have to make, in order to minimise the time we spend
        actually computing samples.

        As such, in this method we determine:
        - How many models we will need to build, by grouping the constraints and the
          causal estimand by the handlers they use.
        - Build these models, returning them in a list called `models`.
        - Build another list that maps the index of components in
          `self._ordered_components` to the index of the model in `models` that they
          use. The causal estimand is by convention the component at index -1 of this
          returned list.

        Args:
            n_samples: Value to be passed to `numpyro.Predictive`'s `num_samples`
                argument for each of the models that are constructed from the underlying
                graph.

        Returns:
            list[Predictive]: List of Predictive models, whose elements contain all the
                models needed by the components.
            list[int]: Mapping of component indexes (as per `self_ordered_components`)
                to the index of the model in the first return argument that the
                component uses.

        """
        models: list[Predictive] = []
        grouped_component_indexes: list[list[int]] = []
        for index, component in enumerate(self._ordered_components):
            # Determine if this constraint uses the same handlers as those of any of
            # the other sets.
            belongs_to_existing_group = False
            for group in grouped_component_indexes:
                # Pull any element from the group to compare models to.
                # Items in a group are known to have the same model, so we can just
                # pull out the first one.
                group_element = self._ordered_components[group[0]]
                # Check if the current constraint can also use this model.
                if component.can_use_same_model_as(group_element):
                    group.append(index)
                    belongs_to_existing_group = True
                    break

            # If the component does not fit into any existing group, create a new
            # group for it. And add the model corresponding to the group to the
            # list of models.
            if not belongs_to_existing_group:
                grouped_component_indexes.append([index])

                models.append(
                    Predictive(
                        component.apply_effects(self._underlying_graph.model),
                        num_samples=n_samples,
                    )
                )

        # Now "invert" the grouping, creating a mapping that maps the index of a
        # component to the (index of the) model it uses.
        component_index_to_model_index = []
        for index in range(len(self._ordered_components)):
            for group_index, group in enumerate(grouped_component_indexes):
                if index in group:
                    component_index_to_model_index.append(group_index)
                    break
        # All indexes should belong to at least one group (worst case scenario,
        # their own individual group). Thus, it is safe to do the above to create
        # the mapping from component index -> model (group) index.
        return models, component_index_to_model_index

    def lagrangian(
        self, n_samples: int = 1000, *, maximum_problem: bool = False
    ) -> Callable[[dict[str, npt.ArrayLike], npt.ArrayLike, jax.Array], npt.ArrayLike]:
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

        The remaining argument of the Lagrangian is the PRNG Key that should be used
        when drawing samples.

        Note that our current implementation assumes there are no equality constraints
        being imposed (in which case, we would need a 3-argument Lagrangian function).

        Args:
            n_samples: The number of random samples to be drawn when estimating the
                value of functions of the RVs.
            maximum_problem: If passed as `True`, assemble the Lagrangian for the
                maximisation problem. Otherwise assemble that for the minimisation
                problem (default behaviour).

        Returns:
            The Lagrangian, as a function of the model parameters, Lagrange multipliers,
                and PRNG key.

        """
        maximisation_prefactor = -1.0 if maximum_problem else 1.0

        # Build association between self.constraints and the model-samples that each
        # one needs to use. We do this here, since once it is constructed, it is
        # FIXED, and doesn't need to be done each time we call the Lagrangian.
        models, component_to_index_mapping = self._associate_models_to_components(
            n_samples
        )

        def _inner(
            parameter_values: dict[str, npt.ArrayLike],
            l_mult: jax.Array,
            rng_key: jax.Array,
        ) -> npt.ArrayLike:
            # Draw samples from all models
            all_samples = tuple(
                sample_model(model, rng_key, parameter_values) for model in models
            )

            value = maximisation_prefactor * self.causal_estimand(all_samples[-1])
            # TODO: https://github.com/UCL/causalprog/issues/87
            value += sum(
                l_mult[i] * c(all_samples[component_to_index_mapping[i]])
                for i, c in enumerate(self.constraints)
            )
            return value

        return _inner
