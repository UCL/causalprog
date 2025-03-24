import distrax
import pytest

from causalprog.distribution.base import SampleTranslator
from causalprog.distribution.family import DistributionFamily


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
    mnv = distrax.MultivariateNormalFullCovariance

    mnv_family = DistributionFamily(mnv, SampleTranslator(rng_key="seed"))
    via_family = mnv_family.construct(
        loc=n_dim_std_normal["mean"], covariance_matrix=n_dim_std_normal["cov"]
    )
    via_backend = mnv(n_dim_std_normal["mean"], n_dim_std_normal["cov"])

    assert via_backend.kl_divergence(via_family.get_dist()) == pytest.approx(0.0)
    assert via_family.get_dist().kl_divergence(via_backend) == pytest.approx(0.0)
