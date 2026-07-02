"""Regression functions (act as constraints for causal problems)."""

from collections.abc import Callable
from typing import TypeAlias

from jax.nn import sigmoid, tanh
from jax.numpy.linalg import norm
from numpy.typing import NDArray

from .graph import ContinuousRandomVariableNode, DiscreteRandomVariableNode, Graph
from .quadrature.base import QuadratureMethod

ModelParam: TypeAlias = dict[str, NDArray]  # Should be dict[str, PyTree] I guess...
MLPAlias: TypeAlias = Callable[[dict[str, NDArray], ModelParam], float | NDArray]


def build_regression_function(
    graph: Graph, theta_x: NDArray, quadrature: QuadratureMethod
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
    node_y: ContinuousRandomVariableNode = graph.get_node("Y")

    def f_x_inverse(xzl: dict[str, NDArray]) -> float | NDArray:
        r"""
        $g(x, z, l) := f_X^{-1}(x, z, l; theta_x).

        At the time of evaluation, this is known (since we are given $\theta_X$).
        """
        return node_ux.evaluate(xzl, {"theta_X": theta_x})

    def pi_ul(ulc: dict[str, NDArray], theta_pi: ModelParam) -> float | NDArray:
        r"""$\pi_ul(c, u, l; theta_pi)."""
        return node_uy.evaluate(ulc, theta_pi)

    def f_y(x_uy: dict[str, NDArray], theta_y: ModelParam) -> float | NDArray:
        r"""$f_Y(x, u_y; theta_y)."""
        return node_y.evaluate(x_uy, theta_y)

    f_m: MLPAlias = node_uy.f_m
    f_r: MLPAlias = node_uy.f_r
    c_values = node_c.possible_values

    def _integrand(
        s_q: float, xzl: dict[str, NDArray], model_params: dict[str, ModelParam]
    ) -> float | NDArray:
        """Regression function integrand, usable with `QuadratureMethod.integrate`."""
        u = f_x_inverse(xzl)

        theta_m = model_params["theta_m"]
        theta_r = model_params["theta_r"]
        theta_pi = model_params["theta_pi"]
        theta_y = model_params["theta_y"]

        x = xzl["X"]
        z = xzl["Z"]
        el = xzl["L"]

        result = 0.0
        for c in c_values:
            czl = {"C": c, "Z": z, "L": el}

            f_r_vector = tanh(f_r(czl, theta_r))
            sigmoid_f_m = sigmoid(f_m(czl, theta_m))
            v_y = 1.0 - sigmoid_f_m**2
            m_y = u * sigmoid_f_m * f_r_vector / (norm(f_r_vector) ** 2)
            u_y = s_q * v_y + m_y

            pi_ul_prediction = pi_ul({"C": c, "L": el, "U_X": u}, theta_pi)
            f_y_prediction = f_y({"X": x, "U_Y": u_y}, theta_y)
            result += pi_ul_prediction * f_y_prediction
        return result

    def _r(
        xzl: dict[str, float | NDArray], model_params: dict[str, NDArray]
    ) -> NDArray:
        r"""
        Regression function, $r$.

        $$ r(x, z, l; theta) = \mathbb{E}[Y \vert X=x, Z=z, L=l]. $$
        """
        return quadrature.integrate(
            _integrand,
            a=-float("inf"),
            b=float("inf"),
            node_values=xzl,
            theta=model_params,
        )

    return _r
