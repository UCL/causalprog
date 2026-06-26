from causalprog.graph import Graph, DataNode, DiscreteRandomVariableNode


def example_model(
    label: str = "G",
    l_len: int = 1,
    z_len: int = 1,
    k: int = 10,
) -> Graph:
    """Create a graph representing the example model.

    Args:
        label: The label of the graph.
        l_len: The number of entries in the vector data node L.
        z_len: The number of entries in the vector data node Z.
        k: The maximum value that could be taken by the mixture indicator C

    Returns:
        A graph
    """
    graph = Graph(label=label)

    graph.add_node(DataNode(label="L", shape=(l_len, )))
    graph.add_node(DataNode(label="Z", shape=(z_len, )))
    graph.add_node(DiscreteRandomVariableNode(label="C", values=list(range(1, k + 1)))

    return graph
