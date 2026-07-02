"""Regression functions (act as constraints for causal problems)."""

from collections.abc import Callable

from jax.nn import sigmoid, tanh
from jax.numpy.linalg import norm
from numpy.typing import NDArray

from .algorithms import evaluate
from .graph import ContinuousRandomVariableNode, DiscreteRandomVariableNode, Graph
from .quadrature.base import QuadratureMethod


def build_regression_function(
    graph: Graph, theta_X: NDArray[float], quadrature: QuadratureMethod
) -> Callable:
    r"""
    Build the regression function for $Y$ given $X, Z, L$.

    Explicitly, the regression function to be constructed is

    $$ r(x, z, l; theta) = \mathbb{E}[Y \vert X=x, Z=z, L=l], $$

    however this can be simplified through our understanding of our
    particular model to be written as

    $$ r(x, z, l; theta) =
    \sum_{q=1}^M w_q \sum_{c=1}^K \pi_{ul}(c)f_{Y}(s_q v_y + m_y, x, l), $$

    where $s_q, w_q$ are sample points drawn from a quadrature rule.

    This function assumes the following (in the context of Ricardo's example graph):
    - $f_X$ (or specifically $\theta_X$) is known, and thus the inverse map
      $g = f^{-1}_X$ is known too. The graph has been suitably edited so that the edge
      connecting $X$ and $U_X$ is now directed _into_ $U_X$.
    - The node $U_Y$ stores the function $\pi_{ul}(c)$ in it's `.compute` attribute.
      $U_Y$ also provides access to the functions $f_r$ and $f_m$ through two of its
      attributes, and has two nodes representing $\theta_r$ and $\theta_m$ as parents.
    """
    node_c: DiscreteRandomVariableNode = graph.get_node("C")
    node_uy: ContinuousRandomVariableNode = graph.get_node("UY")
    node_ux: ContinuousRandomVariableNode = graph.get_node("UX")
    node_x: ContinuousRandomVariableNode = graph.get_node("X")
    node_y: ContinuousRandomVariableNode = graph.get_node("Y")

    f_m = node_uy.f_m
    f_r = node_uy.f_r
    f_pi = node_uy.f_pi  # would be replaced with graph.evaluate("U_Y", ...)
    # Should be using graph.evaluate here!
    # Though given that we'll also want to compute the intermediate values to compute
    # the _v_y, _m_y, etc, it would be ideal if evaluate returned a dict of all nodes
    # that were evaluated...!
    f_y = node_y.f_y  # would be replaced with graph.evaluate("Y", ...)

    c_values = node_c.possible_values

    # _integrand(s_q, node_values, theta_values) instead?
    def _integrand(s_q, z, l, x, theta):
        # NOTE: Could just do graph.get_node("U_X").compute here right? It might even
        # be less effort than a full evaluate of the graph...?
        # Something like u = node_ux.compute({"X": x, "Z": z, "L": l}, theta)
        u = evaluate(graph, "U_X", {"l": l, "z": z, "x": x, "theta_X": theta_X})
        theta_m = theta["theta_m"]
        theta_r = theta["theta_r"]

        result = 0.0
        for c in c_values:
            f_r_vector = tanh(f_r(c, z, l, theta_r))
            sigmoid_f_m = sigmoid(f_m(c, z, l, theta_m))
            v_y = 1.0 - sigmoid_f_m**2
            m_y = u * sigmoid_f_m * f_r_vector / (norm(f_r_vector) ** 2)
            u_y = s_q * v_y + m_y

            # pi_ul = node_uy.compute({"U_X": u, "C": c}, theta) would also work?
            pi_ul = evaluate(
                graph,
                "U_Y",
                {
                    "C": c,
                    "L": l,
                    "Z": z,
                    "X": x,
                    "U_X": u,
                    "theta_X": theta_X,
                    "theta_pi": theta["theta_pi"],
                },
            )
            # This will waste compute time re-calculating everything in the graph that
            # came before, though?
            # Alternative would be f_y = node_y.compute({"U_Y": u_y, "X": x}, theta)
            f_y = evaluate(
                graph,
                "Y",
                {
                    "C": c,
                    "L": l,
                    "Z": z,
                    "X": x,
                    "U_X": u,
                    "U_Y": u_y,
                    "theta_X": theta_X,
                    **theta,
                },
            )

            result += pi_ul * f_y
        return result

    def _r(theta: dict[str, NDArray], x: float, z: float, l: float) -> NDArray:
        """"""
        return quadrature.integrate(
            _integrand, a=-float("inf"), b=float("inf"), z=z, l=l, x=x, theta=theta
        )

    return _r
