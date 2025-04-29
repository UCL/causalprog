import jax.numpy as jnp
import pytest
from distrax import MultivariateNormalFullCovariance as Mvn

from causalprog.backend.translation import Translation
from causalprog.distribution.base import Distribution
from causalprog.distribution.family import DistributionFamily
from causalprog.distribution.normal import Normal, NormalFamily


@pytest.mark.parametrize(
    ("n_dim_std_normal"),
    [pytest.param(1, id="1D normal"), pytest.param(3, id="3D normal")],
    indirect=["n_dim_std_normal"],
)
def test_sampling_consistency(rng_key, n_dim_std_normal) -> None:
    """"""
    sample_shape = (5, 10)
    normal_family = NormalFamily()

    via_family = normal_family.construct(**n_dim_std_normal)
    via_standard_class = Normal(**n_dim_std_normal)

    family_samples = via_family.sample(rng_key, sample_shape)
    standard_class_samples = via_standard_class.sample(rng_key, sample_shape)

    assert jnp.allclose(family_samples, standard_class_samples)


class DistraxNormal(Distribution):
    def __init__(self, mean, cov):
        super().__init__(
            Translation(
                backend_name="sample",
                frontend_name="sample",
                param_map={"seed": "rng_key"},
            ),
            backend=Mvn(mean, cov),
            label="Distrax normal",
        )


@pytest.mark.parametrize(
    ("n_dim_std_normal"),
    [pytest.param(2, id="2D normal")],
    indirect=["n_dim_std_normal"],
)
def test_builder_matches_backend(n_dim_std_normal) -> None:
    """
    Test that building from a family is equivalent
    to building via the backend explicitly.

    """
    mnv_family = DistributionFamily(
        DistraxNormal,
        label="Distrax normal family",
    )
    via_family = mnv_family.construct(**n_dim_std_normal)
    via_backend = Mvn(
        loc=n_dim_std_normal["mean"], covariance_matrix=n_dim_std_normal["cov"]
    )

    assert via_backend.kl_divergence(via_family.dist) == pytest.approx(0.0)
    assert via_family.dist.kl_divergence(via_backend) == pytest.approx(0.0)
