"""Regression functions (act as constraints for causal problems)."""

from collections.abc import Callable

from jax.nn import sigmoid, softmax, tanh
from jax.numpy.linalg import norm
from numpy.typing import NDArray

from .graph import ContinuousRandomVariableNode, DiscreteRandomVariableNode, Graph
from .quadrature.base import QuadratureMethod


def build_regression_function(
    graph: Graph, trained_parameters: dict[str, NDArray], quadrature: QuadratureMethod
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
    """
    node_c: DiscreteRandomVariableNode = graph.get_node("C")
    node_uy: ContinuousRandomVariableNode = graph.get_node("UY")
    node_ux: ContinuousRandomVariableNode = graph.get_node("UX")
    node_x: ContinuousRandomVariableNode = graph.get_node("X")
    node_y: ContinuousRandomVariableNode = graph.get_node("Y")
    # TODO: Need to know where these attributes are stored on the graph nodes.
    # In particular, these f_r and f_m aren't actually going to be at the _func
    # attribute, since that's going to be \f_pi??
    # Also, we could maybe store sigma_czl on $U_Y$ itself, to save assembling it
    # here?
    f_m = node_uy.f_m
    f_r = node_uy.f_r
    f_pi = node_uy.f_pi  # would be replaced with graph.evaluate("U_Y", ...)
    # Should be using graph.evaluate here!
    # Though given that we'll also want to compute the intermediate values to compute
    # the _v_y, _m_y, etc, it would be ideal if evaluate returned a dict of all nodes
    # that were evaluated...!
    f_y = node_y.f_y  # would be replaced with graph.evaluate("Y", ...)

    # Need an inverse step here too? So whatever normalising flow thing is attached
    # to X, it will also need a back-propagation method too?
    # Also, is this going to place nicely with .evaluate??? Since there will be times
    # when we want to fix the value of X=x, and then use g to find the value of U, and
    # other times when we just want to compute the value of X from u, z, l... but then
    # a two-way edge means we don't have a DAG
    def g(x: float, u: float, l: float) -> NDArray:
        return node_x.f_inverse(x, u, l, trained_parameters["theta_x"])

    c_values = node_c.possible_values

    # TODO: c, z, l are all floats... but this is vectorizable potentially?
    def _sigma(
        c: float, z: float, l: float, theta_m: NDArray, theta_r: NDArray
    ) -> NDArray:
        f_r_vector = tanh(f_r(c, z, l, theta_r))
        sigmoid_f_m = sigmoid(f_m(c, z, l, theta_m))

        return (sigmoid_f_m**2) * f_r_vector / (norm(f_r_vector) ** 2)

    def _sigma_squared(c: float, z: float, l: float, theta_m: NDArray) -> float:
        # NOTE: In Ricardo's notes, this is f_r and theta_r inside the sigmoid,
        # but that doesn't make sense if I do the calculation myself? Think it
        # might be a typo
        return sigmoid(f_m(c, z, l, theta_m)) ** 2

    # I think this is the thing that needs to live on the U_Y node.
    # Since it's the compute() function, given c, u, and l.
    def _pi_ul(c: float, u: float, l: float, theta_pi: NDArray) -> NDArray:
        return softmax(f_pi(c, u, l, theta_pi))

    def _v_y(c: float, z: float, l: float, theta_m: NDArray) -> NDArray:
        return 1.0 - _sigma_squared(c, z, l, theta_m)

    def _m_y(
        c: float,
        z: float,
        l: float,
        x: float,
        theta_m: NDArray,
        theta_r: NDArray,
    ) -> NDArray:
        return _sigma(c, z, l, theta_m, theta_r) * g(x, z, l)

    def _integrand(s_q, z, l, x, theta):
        u = g(x, z, l)
        result = 0.0

        for c in c_values:
            result += _pi_ul(c, u, l, theta["theta_pi"]) * f_y(
                s_q * _v_y(c, z, l, theta["theta_m"])
                + _m_y(c, z, l, x, theta["theta_m"], theta["theta_r"]),
                x,
                l,
            )
        return result

    def _r(theta: dict[str, NDArray], x: float, z: float, l: float) -> NDArray:
        """"""
        return quadrature.integrate(
            _integrand, a=-float("inf"), b=float("inf"), z=z, l=l, x=x, theta=theta
        )

    return _r
