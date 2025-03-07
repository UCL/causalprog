import networkx as nx

import causalprog


def test_root_distribution_node_label():
    family = causalprog.graph.node.DistributionFamily()
    node = causalprog.graph.RootDistributionNode(family, "N0")
    node2 = causalprog.graph.RootDistributionNode(family, "N1")
    node3 = causalprog.graph.RootDistributionNode(family, "Y")
    node4 = causalprog.graph.DistributionNode(family, "N4")
    node_copy = node

    assert node.label == node_copy.label
    assert node.label != node2.label
    assert node.label != node3.label
    assert node.label != node4.label

    assert isinstance(node, causalprog.graph.node.Node)
    assert isinstance(node2, causalprog.graph.node.Node)
    assert isinstance(node3, causalprog.graph.node.Node)
    assert isinstance(node4, causalprog.graph.node.Node)


def test_simple_graph():
    family = causalprog.graph.node.DistributionFamily()
    n_x = causalprog.graph.RootDistributionNode(family, "N_X")
    n_m = causalprog.graph.RootDistributionNode(family, "N_M")
    u_y = causalprog.graph.RootDistributionNode(family, "U_Y")
    x = causalprog.graph.DistributionNode(family, "X")
    m = causalprog.graph.DistributionNode(family, "M")
    y = causalprog.graph.DistributionNode(family, "Y", is_outcome=True)

    nx_graph = nx.Graph()
    nx_graph.add_edges_from([[n_x, x], [n_m, m], [u_y, y], [x, m], [m, y]])

    graph = causalprog.graph.Graph(nx_graph, "G0")

    assert graph.label == "G0"
