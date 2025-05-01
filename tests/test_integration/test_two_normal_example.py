""""""

import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize as jax_minimize

from causalprog.algorithms import expectation
from causalprog.causal_problem import CausalProblem
from causalprog.distribution.normal import NormalFamily
from causalprog.graph import DistributionNode, Graph, Node, ParameterNode


def test_two_normal_example(
    rng_key: jax.Array,
    n_samples: int = 1000,
    nu_x: float = 1.0,
    nu_y: float = 1.0,
    data: float = 2.0,
    eps: float = 1.0,
    initial_guess: dict[str, float] = {"mu_x": 1.1},
) -> None:
    """"""
    data = jnp.array(data, ndmin=1)

    mu_x = ParameterNode("mu_x")
    x = DistributionNode(
        NormalFamily(),
        label="x",
        parameters={"mean": "mu_x"},
        constant_parameters={"cov": nu_x**2},
    )
    y = DistributionNode(
        NormalFamily(),
        label="y",
        parameters={"mean": "x"},
        constant_parameters={"cov": nu_y**2},
    )

    graph = Graph(label="G")
    graph.add_edge(mu_x, x)
    graph.add_edge(x, y)

    def sigma(g: Graph, rv: Node):
        return expectation(g, rv.label, samples=n_samples, rng_key=rng_key)

    def constraints(g: Graph, rv: Node):
        return expectation(g, rv.label, samples=n_samples, rng_key=rng_key)

    cp = CausalProblem(graph=graph, label="CP")
    cp.set_causal_estimand(sigma, rvs_to_nodes={"rv": "y"}, graph_argument="g")
    cp.set_constraints(constraints, rvs_to_nodes={"rv": "x"}, graph_argument="g")

    min_value = jax_minimize(
        cp.causal_estimand, cp.parameter_vector, options={"maxiter": 5}, method="BFGS"
    )

    # # scipy doesn't like jax arrays, so we also have to be even more inefficient here.
    # # However, SCIPY does do constrained optimisation which jax currently doesn't?
    # # ``jaxopt`` is recommended by the jax devs, worth a look?

    # fn = lambda p: np.array(cp.causal_estimand(p))

    # # This would ideally be done within the CP class, via a method.
    # # But for now, we do it explicitly.
    # data_constraint = NonlinearConstraint(
    #     lambda x: jnp.abs(cp.constraints(x) - data).__array__(), lb=-jnp.inf, ub=eps
    # )
    # # Initial guess would also need to be set a-prori or an argument to hypothetical
    # # method
    # cp.set_parameter_values(**initial_guess)

    # # Should be able to minimise now?
    # min_result = minimize(
    #     fn,
    #     cp.parameter_vector.__array__(),
    #     # constraints=(data_constraint,),
    #     options={"disp": True},
    # )
