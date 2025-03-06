import causalprog


def test_root_distribution_node_label():
    family = causalprog.graph.node.DistributionFamily()
    node = causalprog.graph.RootDistributionNode(family)
    node2 = causalprog.graph.RootDistributionNode(family)
    node3 = causalprog.graph.RootDistributionNode(family, "Y")
    node_copy = node

    assert node.label == node_copy.label
    assert node.label != node2.label
    assert node.label != node3.label

    assert isinstance(node, causalprog.graph.node.Node)
    assert isinstance(node2, causalprog.graph.node.Node)
    assert isinstance(node3, causalprog.graph.node.Node)
