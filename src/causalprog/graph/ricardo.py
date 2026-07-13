"""Functions to create example graphs."""

from collections.abc import Callable
from typing import Any as TODO
from typing import TypeAlias

from jax.nn import sigmoid, softmax, tanh
from jax.numpy.linalg import norm
from numpy.typing import NDArray

from causalprog.quadrature import UniformWeightMonteCarloGaussianQuadrature as UWMCGQuad
from causalprog.quadrature.base import QuadratureMethod

from .graph import Graph
from .node import ContinuousRandomVariableNode, DataNode, DiscreteRandomVariableNode

ModelParam: TypeAlias = dict[str, NDArray]  # Should be dict[str, PyTree] I guess...
MLPAlias: TypeAlias = Callable[[dict[str, NDArray], ModelParam], float | NDArray]


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


def build_regression_function(
    graph: Graph, theta_x: NDArray, quadrature: QuadratureMethod
) -> MLPAlias:
    r"""
    Build the regression function for $Y$ given $X, Z, L$.

    Explicitly, the regression function to be constructed is

    $$ r(x, z, l; theta) = \mathbb{E}[Y \vert X=x, Z=z, L=l], $$

    however this can be simplified through our understanding of our
    particular model to be written as

    $$ r(x, z, l; theta) =
    \sum_{q=1}^M w_q \sum_{c=1}^K \pi_{ul}(c)f_{Y}(s_q v_y + m_y, x, l), $$

    where $s_q, w_q$ are sample points drawn from a quadrature rule.

    Additionally, note that the callable `r` returned by the method has signature
    `r(xzl, model_params)`, rather than the mathematical $r(x, z, l; theta)$.

    This function assumes the following (in the context of Ricardo's example graph):
    - $f_X$ (or specifically $\theta_X$) is known, and thus the inverse map
      $g = f^{-1}_X$ is known too. The graph has been suitably edited so that the edge
      connecting $X$ and $U_X$ is now directed _into_ $U_X$.
    - The node $U_Y$ stores the function $\pi_{ul}(c)$ in it's `.compute` attribute.
      $U_Y$ also provides access to the functions $f_r$ and $f_m$ through two of its
      attributes, and has two nodes representing $\theta_r$ and $\theta_m$ as parents.
    """
    if not isinstance(quadrature, UWMCGQuad):
        msg = (
            "Only UniformWeightMonteCarloGaussianQuadrature "
            "is supported as a quadrature method."
        )
        raise NotImplementedError(msg)

    node_c: DiscreteRandomVariableNode = graph.get_node("c")
    node_uy: ContinuousRandomVariableNode = graph.get_node("u_y")
    node_ux: ContinuousRandomVariableNode = graph.get_node("u_x")
    node_y: ContinuousRandomVariableNode = graph.get_node("y")

    def f_x_inverse(xzl: dict[str, NDArray]) -> float | NDArray:
        r"""
        $g(x, z, l) := f_X^{-1}(x, z, l; theta_x).

        At the time of evaluation, this is known (since we are given $\theta_X$).
        """
        return node_ux.compute(xzl, theta_x)

    def pi_ul(ulc: dict[str, NDArray], theta_pi: ModelParam) -> NDArray:
        r"""$\pi_ul(c, u, l; theta_pi)."""
        return softmax(node_uy.compute(ulc, theta_pi))

    def f_y(x_uy: dict[str, NDArray], theta_y: ModelParam) -> float | NDArray:
        r"""$f_Y(x, u_y; theta_y)."""
        return node_y.compute(x_uy, theta_y)

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

        x = xzl["x"]
        z = xzl["z"]
        el = xzl["l"]

        result = 0.0
        for i_c, c in enumerate(c_values):
            czl = {"c": c, "z": z, "l": el}

            f_r_vector = tanh(f_r(czl, theta_r))
            sigmoid_f_m = sigmoid(f_m(czl, theta_m))
            v_y = 1.0 - sigmoid_f_m**2
            m_y = u * sigmoid_f_m * f_r_vector / (norm(f_r_vector) ** 2)
            u_y = s_q * v_y + m_y

            pi_ul_prediction = pi_ul({"c": c, "l": el, "u_x": u}, theta_pi)[i_c]
            f_y_prediction = f_y({"x": x, "u_y": u_y}, theta_y)
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
            xzl=xzl,
            model_params=model_params,
        )

    return _r


def learn_initaliser(
    graph: Graph,
    r: MLPAlias,
    phi_hat: NDArray,
    quadrature: QuadratureMethod,
    evaluation_points: dict[str, NDArray],
    *,
    r_hat_pts: NDArray | None = None,
    r_hat: Callable | None = None,
) -> NDArray:
    r"""
    Compute the argmin of the function $B($\theta$)$.

    $$ B(\theta) = \frac{1}{N}\sum_i^N \left( \hat{r}_i - r_i(\theta) \right)^2, $$

    where:

    - $\hat{r}(x, z, l)$ is a learnt estimate of the regression function,
    - $r(x, z, l; \theta)$ is the estimate of the regression function using the graph
        structure,
    - $\theta$ are the model parameters over which to optimise,
    - and the summation is taken over a set of evaluation points
        $\mathcal{D} = \left\{ (x^{(i)}, z^{(i)}, l^{(i)}) \right\}_{i=1}^N$.
        Subscript $i$s denote evaluation at the $i$-th evaluation point.

    The `evaluation_points` ($\mathcal{D}$) should be passed as a dictionary of arrays,
    where slices across the keys of this dictionary correspond to individual evaluation
    points $i$.
    $\hat{r}$ may either be provided as a callable Python object, in which case it
    should have a signature compatible with `jax.vmap` when passed `evaluation_points`,

    Args:
        graph: The causal graph
        r: The function r
        phi_hat: The values for phi computes from the training set
        quadrature: A quadrature method
        evaluation_points: A set of evaluation points
        r_hat_pts: The values of the estimate of r at the evaluation points
        r_hat: Callable function that evaluates r_hat

    """
    if (r_hat_pts is None) == (r_hat is None):
        msg = "Either provide r_hat as a callable, OR as a set of points."
        raise RuntimeError(msg)
    if r_hat_pts is None:
        # To clarify: evaluation points is going to be a "vectorised dictionary" right?
        # IE an input that's compatible with jax.vmap?
        # Or is it going to be an iterable of dict inputs? We need to decide on this!
        r_hat_pts = r_hat(evaluation_points)

    def B(theta: TODO) -> TODO:
        return (
            sum(
                (r_entry - r(evaluation_point)) ** 2
                for r_entry, p in zip(r_hat_pts, evaluation_points, strict=False)
            )
            / evaluation_points.shape[0]
        )

    # Use gradient descent method to find the argmin phi_hat
    return phi_hat
