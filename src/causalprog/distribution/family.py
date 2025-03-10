"""Parametrised groups of ``Distribution``s."""

from collections.abc import Callable
from typing import Generic, TypeVar

from numpy.typing import ArrayLike

from causalprog.distribution.base import Distribution, SupportsSampling
from causalprog.utils.translator import Translator

CreatesDistribution = TypeVar(
    "CreatesDistribution", bound=Callable[..., SupportsSampling]
)


class DistributionFamily(Generic[CreatesDistribution]):
    r"""
    A family of ``Distributions``, that share the same parameters.

    A ``DistributionFamily`` is essentially a ``Distribution`` that has not yet had its
    parameter values explicitly specified. Explicit values for the parameters can be
    passed to a ``DistributionFamily``'s ``construct`` method, which will then proceed
    to construct a ``Distribution`` with those parameter values.

    As an explicit example, the (possibly multivariate) normal distribution is
    parametrised by two quantities - the (vector of) mean values $\mu$ and covariates
    $\Sigma. A ``DistributionFamily`` represents this general $\mathcal{N}(\mu, \Sigma)$
    parametrised form of distributions, however without explicit $\mu$ and $\Sigma$
    values we cannot perform operations like drawing samples. Specifying, for example,
    $\mu = 0$ and $\Sigma = 1$ by invoking ``.construct(0., 1.)`` will return a
    ``Distribution`` instance representing $\mathcal{N}(0., 1.)$, which can then have
    samples drawn from it.
    """

    _family: CreatesDistribution
    _family_translator: Translator

    @property
    def _member(self) -> Callable[..., Distribution]:
        """Constructor method for family members, given parameters."""
        return lambda *parameters: Distribution(
            self._family(*parameters), backend_translator=self._family_translator
        )

    def __init__(
        self,
        backend_family: CreatesDistribution,
        backend_translator: Translator,
    ) -> None:
        """
        Create a new family of distributions.

        Args:
            backend_family (CreatesDistribution): Backend callable that assembles the
                distribution, given explicit parameter values.
            backend_translator (Translator): ``Translator`` instance that to be
                passed to the ``Distribution`` constructor.

        """
        self._family = backend_family
        self._family_translator = backend_translator

    def construct(self, *parameters: ArrayLike) -> Distribution:
        """
        Create a distribution from an explicit set of parameters.

        Args:
            *parameters (ArrayLike): Parameters that define a member of this family,
                passed as sequential arguments.

        """
        return self._member(*parameters)
