"""Extension of the Graph class providing features for solving causal problems."""

from collections.abc import Callable
from inspect import signature
from typing import TypeAlias, TypeVar

import jax
import jax.numpy as jnp

from causalprog._abc.labelled import Labelled

from .graph import Graph
from .node import Node

R = TypeVar("R")
CausalEstimand: TypeAlias = Callable[..., float]
Constraints: TypeAlias = Callable[..., float]


class CausalProblem(Labelled):
    """f."""

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

    @property
    def parameter_values(self) -> jax.Array[float]:
        """Returns the vector of parameter values."""
        return jnp.array(
            tuple(node.value for node in self.graph.parameter_nodes), ndmin=1
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

    def _parameter_vector_to_dict(self, parameter_vector: jax.Array[R]) -> dict[str, R]:
        """
        Convert a vector values to a dictionary mapping labels to parameter values.

        Convention is that a vector of parameter values contains values in the same
        order as self.graph.parameter_nodes.

        TODO: test me!
        """
        # Avoid recomputing the parameter node tuple every time.
        pn = self.graph.parameter_nodes
        return {pn[i].label: value for i, value in enumerate(parameter_vector)}

    def _set_parameters(self, parameter_vector: jax.Array) -> None:
        """Shorthand to set parameter node values from a parameter vector."""
        self.graph.set_parameters(**self._parameter_vector_to_dict(parameter_vector))

    def set_sigma(
        self,
        sigma: CausalEstimand,
        rv_to_nodes: dict[str, str] | None = None,
        graph_argument: str | None = None,
    ) -> None:
        """
        Set the Causal Estimand for this problem.

        `sigma` should be a callable object that defines the Causal Estimand of
        interest, in terms of the random variables of interest to the problem. The
        random variables are in turn represented by `Node`s, with this association being
        recorded in the `rv_to_nodes` dictionary.

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

        TODO: test me!

        """
        self._sigma = sigma
        self._sigma_mapping = {}

        if rv_to_nodes is None:
            rv_to_nodes = {}
        sigma_args = signature(sigma).parameters

        for rv_name, node_label in rv_to_nodes.items():
            if rv_name not in sigma_args:
                msg = f"{rv_name} is not a parameter to causal estimand provided."
                raise ValueError(msg)
            self._sigma_mapping[rv_name] = self.graph.get_node(node_label)

        # Any unaccounted-for RV arguments to sigma are assumed to match
        # the label of the corresponding node.
        args_not_used = set(sigma_args.keys()) - set(self._sigma_mapping.keys())
        for arg in args_not_used:
            self._sigma_mapping[arg] = self.graph.get_node(arg)

        # Temporary hack to ensure that we can use expectation(graph, X) syntax.
        if graph_argument:
            self._sigma_mapping[graph_argument] = self.graph

    def causal_estimand(self, p: jax.Array) -> float:
        """
        Evaluate the Causal Estimand at parameter vector `p`.

        Args:
            p (jax.Array): Vector of parameter values to evaluate at.

        TODO: Test me!

        """
        # Set parameter nodes to their new values.
        self._set_parameters(p)
        # Call stored function with transformed arguments.
        return self._sigma(**self._sigma_mapping)
