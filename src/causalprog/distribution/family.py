"""Parametrised groups of ``Distribution``s."""

from collections.abc import Callable
from typing import Generic, TypeVar

import jax
import numpy as np
from numpy import typing as npt

from causalprog._abc.labelled import Labelled
from causalprog.distribution.base import Distribution, SupportsSampling
from causalprog.utils.translator import Translator

CreatesDistribution = TypeVar(
    "CreatesDistribution", bound=Callable[..., SupportsSampling]
)


class DistributionFamily(Generic[CreatesDistribution], Labelled):
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

    _family: CreatesDistribution
    _family_translator: Translator | None

    @property
    def _member(self) -> Callable[..., Distribution]:
        """Constructor method for family members, given parameters."""
        return lambda **parameters: Distribution(
            self._family(**parameters),
            backend_translator=self._family_translator,
        )

    def __init__(
        self,
        backend_family: CreatesDistribution,
        backend_translator: Translator | None = None,
        *,
        family_name: str = "DistributionFamily",
    ) -> None:
        """
        Create a new family of distributions.

        Args:
            backend_family (CreatesDistribution): Backend callable that assembles the
                distribution, given explicit parameter values. Currently, this callable
                can only accept the parameters as a sequence of positional arguments.
            backend_translator (Translator): ``Translator`` instance that to be
                passed to the ``Distribution`` constructor.

        """
        super().__init__(label=family_name)

        self._family = backend_family
        self._family_translator = backend_translator

    def construct(self, **parameters: npt.ArrayLike) -> Distribution:
        """
        Create a distribution from an explicit set of parameters.

        Args:
            **parameters: Parameters that define a member of this family,
                passed as sequential arguments.

        """
        return self._member(**parameters)

    def sample(
        self,
        samples: int,
        rng_key: jax.Array,
        **kwargs: npt.ArrayLike,
    ) -> npt.NDArray[float]:
        """Sample values from the distribution."""

        output = np.zeros(samples)
        new_key = jax.random.split(rng_key, samples)
        for sample in range(samples):
            parameters = {
                param_name: param_sample[sample] if hasattr(param_sample, "__len__") else param_sample for param_name, param_sample in kwargs.items()
            }
            output[sample] = self.construct(**parameters).sample(new_key[sample], 1)[0][
                0
            ]
        return output
