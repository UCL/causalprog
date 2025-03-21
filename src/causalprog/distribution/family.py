"""Parametrised groups of ``Distribution``s."""

from collections.abc import Callable
from typing import Generic, TypeVar

from numpy.typing import ArrayLike

from causalprog._abc.labelled import Labelled
from causalprog.distribution.base import Distribution, SupportsSampling

GenericDistribution = TypeVar(
    "GenericDistribution", bound=Distribution[SupportsSampling]
)


class DistributionFamily(Generic[GenericDistribution], Labelled):
    r"""
    A family of ``Distributions``, that share the same parameters.

    A ``DistributionFamily`` is essentially a ``Distribution`` that has not yet had its
    parameter values explicitly specified. Explicit values for the parameters can be
    passed to a ``DistributionFamily``'s ``construct`` method, which will then proceed
    to construct a ``Distribution`` with those parameter values.

    As an explicit example, the (possibly multivariate) normal distribution is
    parametrised by two quantities - the (vector of) mean values $\mu$ and covariates
    $\Sigma$. A ``DistributionFamily`` represents this general
    $\mathcal{N}(\mu, \Sigma)$ parametrised form, however without explicit $\mu$ and
    $\Sigma$ values we cannot perform operations like drawing samples. Specifying, for
    example, $\mu = 0$ and $\Sigma = 1$ by invoking ``.construct(0., 1.)`` will return a
    ``Distribution`` instance representing $\mathcal{N}(0., 1.)$, which can then have
    samples drawn from it.
    """

    _family: Callable[..., GenericDistribution]

    def __init__(
        self,
        family: Callable[..., GenericDistribution],
        *,
        label: str,
    ) -> None:
        """
        Create a new family of distributions.

        Args:
            family (Callable[..., GenericDistribution]): Backend callable that assembles
                a member distribution of this family, from explicit parameter values.
            label (str): Name to give to the distribution family.

        """
        super().__init__(label=label)

        self._family = family

    def construct(
        self, *pos_parameters: ArrayLike, **kw_parameters: ArrayLike
    ) -> Distribution:
        """Create a distribution from an explicit set of parameters."""
        return self._family(*pos_parameters, **kw_parameters)
