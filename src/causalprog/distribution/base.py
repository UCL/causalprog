"""Base class for backend-agnostic distributions."""

from abc import abstractmethod
from collections.abc import Callable
from typing import Generic, TypeVar

from numpy.typing import ArrayLike

from causalprog._abc.labelled import Labelled
from causalprog.utils.translator import Translator

SupportsRNG = TypeVar("SupportsRNG")
SupportsSampling = TypeVar("SupportsSampling", bound=object)


class SampleTranslator(Translator):
    """
    Translate methods for sampling from distributions.

    The ``Distribution`` class provides a ``sample`` method, that takes ``rng_key`` and
    ``sample_shape`` as its arguments. Instances of this class transform the these
    arguments to those that a backend distribution expects.
    """

    @property
    def _frontend_method(self) -> str:
        return "sample"

    @property
    def compulsory_frontend_args(self) -> set[str]:
        """Arguments that are required by the frontend function."""
        return {"rng_key", "sample_shape"}


class AbstractDistribution(Generic[SupportsSampling], Labelled):
    """A distribution that can be sampled from."""

    @abstractmethod
    def sample(self, rng_key: SupportsRNG, sample_shape: ArrayLike = (), **kwargs: ArrayLike) -> ArrayLike:
        """
        Draw samples from the distribution.

        Args:
            rng_key (SupportsRNG): Key or seed object to generate random samples.
            sample_shape (ArrayLike): Shape of samples to draw.

        Returns:
            Randomly-drawn samples from the distribution.

        """


class Distribution(AbstractDistribution):
    """A (backend-agnostic) distribution that can be sampled from."""


    _dist: SupportsSampling
    _backend_translator: SampleTranslator

    @property
    def _sample(self) -> Callable[..., ArrayLike]:
        """Method for drawing samples from the backend object."""
        return getattr(self._dist, self._backend_translator.backend_method)

    def __init__(
        self,
        backend_distribution: SupportsSampling,
        backend_translator: SampleTranslator | None = None,
        *,
        label: str = "Distribution",
    ) -> None:
        """
        Create a new Distribution.

        Args:
            backend_distribution (SupportsSampling): Backend object that supports
                drawing random samples.
            backend_translator (SampleTranslator): Translator object mapping backend
                sampling function to frontend arguments.

        """
        super().__init__(label=label)

        self._dist = backend_distribution

        # Setup sampling calls, and perform one-time check for compatibility
        self._backend_translator = (
            backend_translator if backend_translator is not None else SampleTranslator()
        )
        self._backend_translator.validate_compatible(backend_distribution)

    def sample(self, rng_key: SupportsRNG, sample_shape: ArrayLike = (), **kwargs: ArrayLike) -> ArrayLike:
        """
        Draw samples from the distribution.

        Args:
            rng_key (SupportsRNG): Key or seed object to generate random samples.
            sample_shape (ArrayLike): Shape of samples to draw.

        Returns:
            Randomly-drawn samples from the distribution.

        """
        args_to_backend = self._backend_translator.translate_args(
            rng_key=rng_key, sample_shape=sample_shape, **kwargs
        )
        return self._sample(**args_to_backend)
        if len(kwargs) == 0:
            return self._dist.sample(rng_key, sample_shape)

        output = np.zeros(samples)
        new_key = jax.random.split(rng_key, samples)
        for sample in range(samples):
            inputs = {i: j.get(sample, j) for i, j in kwargs}
            output[sample] = self._dist.sample(new_key[sample], 1, **inputs)[0][0]
        return output
