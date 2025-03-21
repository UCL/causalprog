"""Base class for backend-agnostic distributions."""

from typing import TypeVar

from numpy.typing import ArrayLike

from causalprog._abc.labelled import Labelled
from causalprog.backend.translation import Translation
from causalprog.backend.translator import Translator

SupportsRNG = TypeVar("SupportsRNG")
SupportsSampling = TypeVar("SupportsSampling", bound=object)


class Distribution(Translator[SupportsSampling], Labelled):
    """A (backend-agnostic) distribution that can be sampled from."""

    @property
    def _frontend_provides(self) -> tuple[str, ...]:
        return ("sample",)

    @property
    def dist(self) -> SupportsSampling:
        """Return the object representing the distribution."""
        return self.get_backend()

    def __init__(
        self, *translations: Translation, backend: SupportsSampling, label: str
    ) -> None:
        """
        Create a new distribution, with a given backend.

        Args:
            *translations (Translation): Information for mapping the methods of the
                backend object to the frontend methods provided by this class. See
                ``causalprog.backend.Translator`` for more details.
            backend (SupportsSampling): Backend object that represents the distribution.
            label (str): Name or label to attach to the distribution.

        """
        Labelled.__init__(self, label=label)
        Translator.__init__(self, *translations, backend=backend)

    def sample(self, rng_key: SupportsRNG, sample_shape: ArrayLike = ()) -> ArrayLike:
        """
        Draw samples from the distribution.

        Args:
            rng_key (SupportsRNG): Key or seed object to generate random samples.
            sample_shape (ArrayLike): Shape of samples to draw.

        Returns:
            ArrayLike: Randomly-drawn samples from the distribution.

        """
        return self._call_backend_with("sample", rng_key, sample_shape)


class NativeDistribution(Distribution[SupportsSampling]):
    """
    A distribution that uses our native backend.

    These distributions do not require translations, since the backend objects
    they use conform to our frontend syntax by design.
    """

    def __init__(self, *, backend: SupportsSampling, label: str) -> None:
        """
        Create a new distribution, using a native backend.

        Args:
            backend (SupportsSampling): Backend object that represents the distribution.
                Must be a native backend object; that is a distribution provided by the
                ``causalprog`` package.
            label (str): Name or label to attach to the distribution.

        """
        super().__init__(backend=backend, label=label)
