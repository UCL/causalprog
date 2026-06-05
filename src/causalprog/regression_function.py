"""Regression functions (act as constraints for causal problems)."""

from numpy.typing import NDArray
from jax.nn import sigmoid, tanh
from jax.numpy.linalg import norm

from .graph import Graph
from .quadrature.base import QuadratureMethod


def build_regression_function(
    graph: Graph, trained_parameters: dict[str, NDArray], quadrature: QuadratureMethod
) -> None:
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
    # attribute, since that's going to be \pi_ul??
    # Also, we could maybe store sigma_czl on $U_Y$ itself, to save assembling it
    # here?
    f_m = graph.get_node("U_Y").f_m
    f_r = graph.get_node("U_Y").f_r
    pi_ul = graph.get_node("U_Y").pi_ul

    # TODO: c, z, l are all floats... but this is vectorizable potentially?
    def _sigma(
        c: float, z: float, l: float, theta_m: NDArray, theta_r: NDArray
    ) -> NDArray:
        f_r_vector = tanh(f_r(c, z, l, theta_r))
        sigmoid_f_m = sigmoid(f_m(c, z, l, theta_m))

        return (sigmoid_f_m**2) * f_r_vector / (norm(f_r_vector) ** 2)

    def _sigma_squared(
        c: float,
        z: float,
        l: float,
        theta_m: NDArray,
    ) -> float:
        # NOTE: In Ricardo's notes, this is f_r and theta_r inside the sigmoid,
        # but that doesn't make sense if I do the calculation myself? Think it
        # might be a typo
        return sigmoid(f_m(c, z, l, theta_m)) ** 2

    def _r(theta, x, z, l) -> NDArray:
        """"""

        return

    return _r
