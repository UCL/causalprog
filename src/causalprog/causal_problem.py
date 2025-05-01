"""Extension of the Graph class providing features for solving causal problems."""

from collections.abc import Callable
from inspect import signature
from typing import TypeAlias

import jax
import jax.numpy as jnp

from causalprog._abc.labelled import Labelled
from causalprog.graph import Graph, Node

CausalEstimand: TypeAlias = Callable[..., float]
Constraints: TypeAlias = Callable[..., float]


def raises(exception: Exception) -> Callable[[], float]:
    """Create a callable that raises ``exception`` when called."""

    def _inner() -> float:
        raise exception

    return _inner


class CausalProblem(Labelled):
    """
    Container class for handling a causal problem.

    A causal problem <https://github-pages.ucl.ac.uk/causalprog/theory/mathematical-context/>
    requires an underlying ``Graph`` to describe the relationships between the random
    variables and parameters, plus a causal estimand and list of (data) constraints.
    Structural constraints are handled by imposing restrictions on forms of the random
    variables and constraints directly.

    A ``CausalProblem`` instance brings together these components, providing a container
    for a causal problem that can be given inputs like empirical data, a solver
    tolerance, etc, and will provide (estimates of) the bounds for the causal estimand.

    - The ``.graph`` attribute stores the underlying ``Graph`` object.
    - The ``.causal_estimand`` method evaluates the causal estimand, given values for
        the parameters.
    - The ``.constraints`` method evaluates the (vector-valued) constraints, given
        values for the parameters.

    The user must specify each of the above before a ``CausalProblem`` can be solved.
    The primary way for this to be done is to construct or load the corresponding
    ``Graph``, and provide it by setting the ``CausalProblem.graph`` attribute directly.
    Then, `set_causal_estimand` and `set_constraints` can be used to provide the causal
    estimand and constraints functions, in terms of the random variables. The
    ``CausalProblem`` instance will handle turning them into functions of the parameter
    values under the hood. Initial parameter values (for the purposes of solving) can be
    provided to the solver method directly or set beforehand via ``set_parameters``. It
    should never be necessary for the user to interact with, or provide, a vector of
    parameters (as this is taken care of under the hood).
    """

    _graph: Graph | None
    _sigma: CausalEstimand
    _sigma_mapping: dict[str, Node]
    _constraints: Constraints
    _constraints_mapping: dict[str, Node]

    @property
    def graph(self) -> Graph:
        """Graph defining the structure of the `CausalProblem`."""
        if self._graph is None:
            msg = f"No graph set for {self.label}."
            raise ValueError(msg)
        return self._graph

    @graph.setter
    def graph(self, new_graph: Graph) -> None:
        if not isinstance(new_graph, Graph):
            msg = f"{self.label}.graph must be a Graph instance."
            raise TypeError(msg)
        self._graph = new_graph

    @property
    def parameter_values(self) -> dict[str, float]:
        """Dictionary mapping parameter labels to their (current) values."""
        return self._parameter_vector_to_dict(self.parameter_vector)

    @property
    def parameter_vector(self) -> jax.Array:
        """Returns the (current) vector of parameter values."""
        return jnp.array(
            tuple(
                node.value if node.value is not None else float("NaN")
                for node in self.graph.parameter_nodes
            ),
            ndmin=1,
        )

    def __init__(
        self,
        graph: Graph | None = None,
        *,
        label: str = "CausalProblem",
    ) -> None:
        """Set up a new CausalProblem."""
        super().__init__(label=label)

        self._graph = graph

        self._sigma = raises(
            NotImplementedError(f"Causal estimand not set for {self.label}.")
        )
        self._sigma_mapping = {}

    def _parameter_vector_to_dict(
        self, parameter_vector: jax.Array
    ) -> dict[str, float]:
        """
        Convert a parameter vector to a dictionary mapping labels to parameter values.

        Convention is that a vector of parameter values contains values in the same
        order as self.graph.parameter_nodes.
        """
        # Avoid recomputing the parameter node tuple every time.
        pn = self.graph.parameter_nodes
        return {pn[i].label: value for i, value in enumerate(parameter_vector)}

    def _set_parameters_via_vector(self, parameter_vector: jax.Array | None) -> None:
        """
        Shorthand to set parameter node values from a parameter vector.

        No intended for frontend use - primary use will be internal when running
        optimisation methods over the CausalProblem, when we need to treat the
        parameters as a vector or array of function inputs.
        """
        self.graph.set_parameters(**self._parameter_vector_to_dict(parameter_vector))

    def set_parameter_values(self, **parameter_values: float | None) -> None:
        """
        Set (initial) parameter values for this CausalProblem.

        See ``Graph.set_parameters`` for input details.
        """
        self.graph.set_parameters(**parameter_values)

    def set_causal_estimand(
        self,
        sigma: CausalEstimand,
        rvs_to_nodes: dict[str, str] | None = None,
        graph_argument: str | None = None,
    ) -> None:
        """
        Set the Causal Estimand for this problem.

        `sigma` should be a callable object that defines the Causal Estimand of
        interest, in terms of the random variables of to the problem. The
        random variables are in turn represented by `Node`s, with this association being
        recorded in the `rv_to_nodes` dictionary.

        The `causal_estimand` method of the instance will be usable once this method
        completes.

        Args:
            sigma (CausalEstimand): Callable object that evaluates the causal estimand
                of interest for this `CausalProblem`, in terms of the random variables,
                which are the arguments to this callable. `sigma`s with additional
                arguments are not currently supported.
            rvs_to_nodes (dict[str, str]): Mapping of random variable (argument) names
                of `sigma` to the labels of the corresponding `Node`s representing the
                random variables. Argument names that match their corresponding `Node`
                label can be omitted.
            graph_argument (str): Argument to `sigma` that should be replaced with
                `self.graph`. This argument is only temporary, as we are currently
                limited to the syntax `expectation(Graph, Node)` rather than just
                `expectation(Node)`. It will be removed in the future when methods like
                `expectation` can be called solely on `Node` objects.

        """
        self._sigma = sigma
        self._sigma_mapping = {}

        if rvs_to_nodes is None:
            rvs_to_nodes = {}
        sigma_args = signature(sigma).parameters

        for rv_name, node_label in rvs_to_nodes.items():
            if rv_name not in sigma_args:
                msg = f"{rv_name} is not a parameter to causal estimand provided."
                raise ValueError(msg)
            self._sigma_mapping[rv_name] = self.graph.get_node(node_label)

        # Any unaccounted-for RV arguments to sigma are assumed to match
        # the label of the corresponding node.
        args_not_used = set(sigma_args.keys()) - set(self._sigma_mapping.keys())

        ## Temporary hack to ensure that we can use expectation(graph, X) syntax.
        if graph_argument:
            self._sigma_mapping[graph_argument] = self.graph
            args_not_used -= {graph_argument}
        ## END HACK
        for arg in args_not_used:
            self._sigma_mapping[arg] = self.graph.get_node(arg)

    def causal_estimand(self, p: jax.Array) -> float:
        """
        Evaluate the Causal Estimand at parameter vector `p`.

        Args:
            p (jax.Array): Vector of parameter values to evaluate at.

        """
        # Set parameter nodes to their new values.
        self._set_parameters_via_vector(p)
        # Call stored function with transformed arguments.
        return self._sigma(**self._sigma_mapping)
