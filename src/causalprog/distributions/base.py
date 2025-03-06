"""Base class for backend-agnostic distributions."""

from collections.abc import Callable
from typing import Generic, NamedTuple, TypeVar

from numpy.typing import ArrayLike

SupportsRNG = TypeVar("SupportsRNG")
SupportsSampling = TypeVar("SupportsSampling")


class SampleInfo(NamedTuple):
    """
    Container class for backend-compatibility.

    Different backends have different syntax for drawing samples from the distributions
    they support. In order to map these different syntaxes to our backend-agnostic
    framework, we need a container class to map the names we have chosen for our
    frontend methods to those used by their corresponding backend method.
    """

    method: str = "sample"
    rng_key: str = "key"
    sample_shape: str = "sample_shape"


class Distribution(Generic[SupportsSampling]):
    """Distributions."""

    _dist: SupportsSampling
    _backend_sample_info: SampleInfo

    @property
    def _sample(self) -> Callable[[SupportsRNG, ArrayLike], ArrayLike]:
        backend_sample_method = getattr(self._dist, self._backend_sample_info.method)
        return lambda key, sample_size: backend_sample_method(
            **{
                self._backend_sample_info.rng_key: key,
                self._backend_sample_info.sample_shape: sample_size,
            }
        )

    def __init__(
        self,
        backend_distribution: SupportsSampling,
        backend_sample_method: str = "sample",
        backend_sample_key_arg: str = "key",
        backend_sample_size_arg: str = "sample_shape",
    ) -> None:
        """Create a new Distribution."""
        self._dist = backend_distribution
        self._backend_sample_info = SampleInfo(
            method=backend_sample_method,
            rng_key=backend_sample_key_arg,
            sample_shape=backend_sample_size_arg,
        )

    def sample(self, rng_key: SupportsRNG, sample_shape: ArrayLike = ()) -> ArrayLike:
        """Draw samples from the distribution."""
        return self._sample(rng_key, sample_shape)
