"""Regression functions (act as constraints for causal problems)."""

from collections.abc import Callable

from jax.nn import sigmoid, softmax, tanh
from jax.numpy.linalg import norm
from numpy.typing import NDArray

from .graph import Graph
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
    # TODO: Need to know where these attributes are stored on the graph nodes.
    # In particular, these f_r and f_m aren't actually going to be at the _func
    # attribute, since that's going to be \f_pi??
    # Also, we could maybe store sigma_czl on $U_Y$ itself, to save assembling it
    # here?
    f_m = graph.get_node("U_Y").f_m
    f_r = graph.get_node("U_Y").f_r
    f_pi = graph.get_node("U_Y").f_pi
    f_y = graph.get_node("Y").f_y

    # Need an inverse step here too? So whatever normalising flow thing is attached
    # to X, it will also need a back-propagation method too?
    def g(x: float, u: float, l: float) -> NDArray:
        return graph.get_node("X").f_inverse(x, u, l, trained_parameters["theta_x"])

    # Need to be able to extract the unique values of the indicator C
    c_values = graph.get_node("C").discrete_values

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
        theta_x: NDArray,
    ) -> NDArray:
        return _sigma(c, z, l, theta_m, theta_r) * g(x, z, l, theta_x)

    def _r(theta: dict[str, NDArray], x: float, z: float, l: float) -> NDArray:
        """"""
        u = g(x, z, l)

        # Naive implementation that I know works
        result = 0.0
        for w_q, s_q in zip(*quadrature.points_and_weights(), strict=True):
            inner_sum = 0.0
            for c in c_values:
                inner_sum += _pi_ul(c, u, l, theta["theta_pi"]) * f_y(
                    s_q * _v_y(c, z, l, theta["theta_m"])
                    + _m_y(c, z, l, x, theta["theta_m"], theta["theta_r"]),
                    x,
                    l,
                )
            result += w_q * inner_sum

        return result

    return _r
