"""Extension of the Graph class providing features for solving causal problems."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from .graph import Graph

if TYPE_CHECKING:
    from .node import ParameterNode


class CausalProblem(Graph):
    """f."""

    _sigma: Callable[..., float]
    _constraints: Callable[..., float]

    @property
    def parameter_values(self) -> tuple[float, ...]:
        """Returns the current parameter values stored for the Causal Problem."""
        return tuple(node.current_value for node in self.parameter_nodes)

    def __init__(self, label: str) -> None:
        """Set up a new CausalProblem."""
        super().__init__(label)

        self.reset_parameters()

    def _call_callable_attribute(
        self, which: Literal["sigma", "constraints"], *parameter_values: float
    ) -> float:
        """
        Evaluate the causal estimand or the constraints function.

        parameter_values should be passed in the order they appear in
        self.parameter_nodes.
        """
        # Set parameter value as per the inputs.
        # Order of *parameter_values is assumed to match the order of
        # self.parameter_nodes.
        self.set_parameters(
            **{
                self.parameter_nodes[i].label: value
                for i, value in enumerate(parameter_values)
            }
        )
        # Call underlying function
        return getattr(self, f"_{which}")()

    def _set_callable_attribute(
        self,
        which: Literal["sigma", "constraints"],
        fn: Callable[..., float],
        name_map: dict[str, str],
    ) -> None:
        """
        Set either the causal estimand (sigma) or constraints function.

        Input ``fn`` is assumed to take random variables as arguments. These are
        transformed, via the ``name_map``, into the corresponding ``Node``s in the
        ``Graph`` describing this causal problem.
        """
        setattr(
            self,
            f"_{which}",
            lambda: fn(
                **{
                    rv_name: self.get_node(node_name)
                    for rv_name, node_name in name_map.items()
                }
            ),
        )

    def reset_parameters(self) -> None:
        """Clear all current values of parameter nodes."""
        self.set_parameters(**{node.label: None for node in self.parameter_nodes})

    def set_parameters(self, **parameter_values: float | None) -> None:
        """
        Set the current value of all parameter nodes to the new values.

        Parameter nodes are identified by variable name. Absent parameters retain their
        current value.
        """
        for name, new_value in parameter_values.items():
            node: ParameterNode = self.get_node(name)
            node.current_value = new_value

    def set_causal_estimand(
        self, sigma: Callable[..., float], rvs_to_nodes: dict[str, str]
    ) -> None:
        """Set the causal estimand of this CausalProblem."""
        self._set_callable_attribute("sigma", sigma, rvs_to_nodes)

    def set_constraints(
        self, constraints: Callable[..., float], rvs_to_nodes: dict[str, str]
    ) -> None:
        """Set the constraints of this CausalProblem."""
        self._set_callable_attribute("constraints", constraints, rvs_to_nodes)

    def causal_estimand(self, *parameter_values: float) -> float:
        """Evaluate the causal estimand."""
        return self._call_callable_attribute("sigma", *parameter_values)

    def constraints(self, *parameter_values: float) -> float:
        """Evaluate the constraints function."""
        return self._call_callable_attribute("constraints", *parameter_values)
