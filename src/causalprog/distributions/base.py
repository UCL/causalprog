"""Base class for backend-agnostic distributions."""

from collections.abc import Callable
from typing import Generic, ParamSpec, TypeVar

from numpy.typing import ArrayLike

RNGKey = TypeVar("RNGKey")
SupportsRNG = TypeVar("SupportsRNG")
SupportsSampling = TypeVar("SupportsSampling")
P = ParamSpec("P")


class Distribution(Generic[SupportsSampling]):
    """Distributions."""

    _dist: SupportsSampling
    _backend_sample_info: dict[
        str, str
    ]  # Should probably make immutable, or into a namedtuple, for example

    @property
    def _sample(self) -> Callable[[RNGKey, ArrayLike], ArrayLike]:
        backend_sample_method = getattr(self._dist, self._backend_sample_info["method"])
        return lambda key, sample_size: backend_sample_method(
            **{
                self._backend_sample_info["key"]: key,
                self._backend_sample_info["sample_size"]: sample_size,
            }
        )

    def __init__(
        self,
        backend_distribution: SupportsSampling,
        backend_sample_function: str = "sample",
        backend_sample_key_arg: str = "key",
        backend_sample_size_arg: str = "sample_size",
    ) -> None:
        """Create a new Distribution."""
        self._dist = backend_distribution
        self._backend_sample_info = {
            "method": backend_sample_function,
            "key": backend_sample_key_arg,
            "sample_size": backend_sample_size_arg,
        }

    def sample(self, key: SupportsRNG, sample_size: ArrayLike = ()) -> ArrayLike:
        """Draw samples from the distribution."""
        return self._sample(key, sample_size)


class DistributionFamily(Generic[SupportsSampling, P]):
    """
    A family of distributions, specified by common parameters.

    Essentially a factory class for `Distribution`s.
    """

    _constructor: Callable[P, SupportsSampling]

    def __init__(self, distribution_constructor: Callable[P, SupportsSampling]) -> None:
        """Specify a family of Distributions."""
        self._constructor = distribution_constructor

    def __call__(self, *args, **kwargs) -> Distribution:  # noqa: ANN002, ANN003
        """Create a Distribution from parameters."""
        return Distribution(self._constructor(*args, **kwargs))
