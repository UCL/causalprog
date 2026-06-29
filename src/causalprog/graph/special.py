"""Functions to create example graphs."""

from collections.abc import Callable

from causalprog.graph import (
    ContinuousRandomVariableNode,
    DataNode,
    DiscreteRandomVariableNode,
    Graph,
)


def example_model(
    *,
    label: str = "example_model",
    l_len: int = 1,
    z_len: int = 1,
    k: int = 10,
    compute_u_x: Callable,
    compute_u_y: Callable,
    compute_phi_x: Callable,
    compute_x: Callable,
    compute_y: Callable,
) -> Graph:
    """
    Create a graph representing the example model.

    Args:
        label: The label of the graph.
        l_len: The number of entries in the vector data node l.
        z_len: The number of entries in the vector data node z.
        k: The maximum value that could be taken by the mixture indicator c.
        compute_u_x: Compute u_x given the value of c.
        compute_u_y: Compute u_y given the value of c.
        compute_phi_x: Compute phi_x given the value of l.
        compute_x: Compute x given the values of z, phi_x and u_x.
        compute_y: Compute x given the values of x and u_y.

    Returns:
        A graph

    """
    graph = Graph(label=label)

    graph.add_node(DataNode(label="l", shape=(l_len,)))
    graph.add_node(DataNode(label="z", shape=(z_len,)))
    graph.add_node(
        DiscreteRandomVariableNode(
            label="c", values=[float(i) for i in range(1, k + 1)]
        )
    )
    graph.add_node(
        ContinuousRandomVariableNode(label="u_x", compute=compute_u_x, parents=["c"])
    )
    graph.add_node(
        ContinuousRandomVariableNode(label="u_y", compute=compute_u_y, parents=["c"])
    )
    graph.add_node(
        ContinuousRandomVariableNode(
            label="phi_x", compute=compute_phi_x, parents=["l"]
        )
    )
    graph.add_node(
        ContinuousRandomVariableNode(
            label="x", compute=compute_x, parents=["z", "phi_x", "u_x"]
        )
    )
    graph.add_node(
        ContinuousRandomVariableNode(label="y", compute=compute_y, parents=["x", "u_y"])
    )

    return graph
