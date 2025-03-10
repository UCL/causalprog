from typing import Callable, TypeVar

from numpy.typing import ArrayLike

from causalprog.distribution.base import Distribution, SupportsSampling
from causalprog.utils.translator import Translator

CreatesDistribution = TypeVar(
    "CreatesDistribution", bound=Callable[..., SupportsSampling]
)


class DistributionFamily:
    _family: CreatesDistribution
    _family_translator: Translator

    def __init__(
        self, backend_family: CreatesDistribution, backend_translator: Translator
    ) -> None:
        self._family = backend_family
        self._family_translator = backend_translator

    def construct(self, *parameters: ArrayLike) -> Distribution:
        return Distribution(self._family(*parameters))
