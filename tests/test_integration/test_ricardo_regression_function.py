import jax
import jax.numpy as jnp
import pytest

from causalprog.graph import Graph
from causalprog.graph.ricardo import build_regression_function, example_model
from causalprog.quadrature import UniformWeightMonteCarloGaussianQuadrature as UWMCGQ

# Need to write suitable compute functions!
# Also need to reverse the x -> u_x edge too.
# u_x needs a compute function.
# u_y needs a compute function from c, u_x, and l (?)
# y needs a compute function from u_y and x (and other vars?)
# Nothing else _needs_ a compute function.
# Then the u_y node needs some extra functions tacked onto it...
# and some new nodes added as parents for f_r, f_m etc.


@pytest.fixture
def z_len() -> int:
    return 5


@pytest.fixture
def k_len() -> int:
    return 10


def test_what_am_i_doing(
    k_len: int, z_len: int, rng_key, n_points: int = 10000
) -> None:
    def f_ux(xzl, theta_x):
        return xzl["x"] * theta_x[0]

    def f_pi(ucl, theta_pi):
        return jnp.ones((k_len,))

    def f_y(xu_y, theta_y):
        return theta_y[0] * jnp.exp(-(xu_y["x"] ** 2))

    f_r = lambda *args, **kwargs: jnp.ones((z_len,))
    f_m = lambda *args, **kwargs: -float("inf")

    g: Graph = example_model(
        k=k_len,
        z_len=z_len,
        compute_u_x=f_ux,
        compute_u_y=f_pi,
        compute_phi_x=None,
        compute_x=None,
        compute_y=f_y,
    )
    # Manually attach methods to node for now
    g.get_node("u_y").f_r = f_r
    g.get_node("u_y").f_m = f_m

    given_theta_x = jnp.ones((1,))
    r = build_regression_function(g, given_theta_x, UWMCGQ(n_points, rng_key=rng_key))
    dr = jax.grad(r, argnums=1)

    # In this setup, we should have
    # r(x, z, l; theta) = theta_y[0]e^{-x^2} and thus
    # dr(x, z, l; theta)/dtheta = 0 except for the theta_y component, which should be
    # dr(x, z, l; theta)/dtheta_y = e^{-x^2}
    r_analytic = lambda xzl, theta: theta["theta_y"][0] * jnp.exp(-(xzl["x"] ** 2))
    dr_analytic = lambda xzl, theta: jnp.exp(-(xzl["x"] ** 2))

    # Create "grid" across which to evaluate the built functions
    num = 20
    theta = {
        "theta_m": jnp.linspace(-1.0, 1.0, num=20).reshape(num, 1),
        "theta_r": jnp.linspace(-1.0, 1.0, num=20).reshape(num, 1),
        "theta_pi": jnp.linspace(-1.0, 1.0, num=20).reshape(num, 1),
        "theta_y": jnp.linspace(-1.0, 1.0, num=20).reshape(num, 1),
    }
    xzl = {
        "x": jnp.linspace(-1.0, 1.0, num=20).reshape(num, 1),
        "z": jnp.linspace(-1.0, 1.0, num=20).reshape(num, 1),
        "l": jnp.linspace(-1.0, 1.0, num=20).reshape(num, 1),
    }

    # This "vectorises" our error functions over all inputs.
    # It's hideous, I know, but it is what it is.
    for key in xzl:  # [*xzl.keys(), *theta.keys()]:
        r = jax.vmap(
            r,
            in_axes=(
                {k: None if k != key else 0 for k in xzl},
                {k: None if k != key else 0 for k in theta},
            ),
        )
        r_analytic = jax.vmap(
            r_analytic,
            in_axes=(
                {k: None if k != key else 0 for k in xzl},
                {k: None if k != key else 0 for k in theta},
            ),
        )
        error_dr = jax.vmap(
            error_dr,
            in_axes=(
                {k: None if k != key else 0 for k in xzl},
                {k: None if k != key else 0 for k in theta},
            ),
        )

    # Evaluate the vectorised functions to check if they really do match?
    assert jnp.allclose(error_r(xzl, theta), 0.0)
    assert jnp.allclose(error_dr(xzl, theta), 0.0)
    pass
