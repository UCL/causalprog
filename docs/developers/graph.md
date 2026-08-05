# Graphs in causalprog

In causalprog, directed acyclic graphs (DAGs) are used to represent causal problems.
This documentation page describes how these graphs are implemented, and is aimed
at developers of the library. Documentation for users can be found in [the documentation
for users](../users/graph.md).

## Graphs

Graphs in causalprog are internally stored as [networkx](https://networkx.org/) graphs, with nodes
being instances of subclasses of `causalprog.graph.base.Node`. The interface to networkx is hidden
from users, with methods defined in the node and graph classes making the direct calls to
networkx. This should make it easier to replace networkx with another graph library in future
if this is desired.

All graph and node classes in causalprog inherit from `causalprog._abc.labelled.Labelled` which
enforces that each instance has a label set at the point of initialisation.

## Nodes

All graph nodes in causalprog must inheret from the `causalprog.graph.base.Node` base class. This
class has the following abstract methods that must be implemented:

- `evaluate` returns an evaluation of the node given values of its parents.
- `copy` makes a (deep) copy of the node.
- `parents` returns the node's parents.
  `parents` is a property instead of a method.
- `sample` samples value(s) from the node.
  This function is no longer used in the examples and could
  be considered for removal.

Inside the initialiser of any subclass of `causalprog.graph.base.Node`, the `super()` initialiser
function must be called, with `label` given as a required keyword argument and `shape` as an
optional second keyword argument, defaulting to `()` for a scalar.

## Algorithms

In general, functions that act on a single node or return information about the full graph are
implemented as methods or properties of the graph or node classes, while functions that iterate
through all nodes in a graph, copy and / or modify it, or are more computationally involved are
implemented as functions in `causalprog.algorithms`.

### Iterating through graphs

The graph algorithms in causalprog typically iterate through a graph starting from the roots of
moving towards the trees. This ordering of nodes can be obtained using the property `ordered_nodes`
or method `roots_down_to_outcome` of the graph - the first of these includes all of the nodes in
the graph while the latter only includes a chosen node and its predecessors.
