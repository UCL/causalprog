"""Base class for backend-agnostic distributions."""

import inspect
from collections.abc import Callable
from typing import Generic, NamedTuple, TypeVar

from numpy.typing import ArrayLike

SupportsRNG = TypeVar("SupportsRNG")
SupportsSampling = TypeVar("SupportsSampling", bound=object)


# Name is not very informative, and this could be more general.
# EG Translator:
# - store name of backend method that we want to call
# - store the frontend variable names that we want to use
# - store mapping of names of backend args -> names of the frontend args
class SampleCompatibility(NamedTuple):
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

    def validate_compatible(self, obj: object) -> None:
        """
        Determine if ``object`` provides a compatible ``sample`` method.

        The ``object`` must provide a callable whose name matches ``self.method``,
        and the callable referenced must take two arguments whose names match
        ``self.rng_key`` and ``self.sample_shape``. This ensures that the information
        stored in the instance will result in a successful call to the referenced
        callable when needed.

        Args:
            obj (type): Object to check possesses a method that can be called with the
                information stored.

        """
        # Check that obj does provide a method of matching name
        if not hasattr(obj, self.method):
            msg = f"Distribution-defining object {obj} has no method '{self.method}'."
            raise AttributeError(msg)
        if not callable(getattr(obj, self.method)):
            msg = f"'{self.method}' attribute of {obj} is not callable."
            raise TypeError(msg)

        # Check that this method will be callable with the information given.
        must_take_args = {self.rng_key, self.sample_shape}
        method_params = inspect.signature(getattr(obj, self.method)).parameters
        # The arguments that will be passed are actually taken by the method.
        for compulsory_arg in must_take_args:
            if compulsory_arg not in method_params:
                msg = f"'{self.method}' does not take argument '{compulsory_arg}'."
                raise TypeError(msg)
        # The method does not _require_ any additional arguments
        method_requires = {
            name for name, p in method_params.items() if p.default is p.empty
        }
        if not method_requires.issubset(must_take_args):
            args_not_accounted_for = method_requires - must_take_args
            raise TypeError(
                f"'{self.method}' not provided compulsory arguments "
                "(missing " + ", ".join(args_not_accounted_for) + ")"
            )


class Distribution(Generic[SupportsSampling]):
    """A (backend-agnostic) distribution that can be sampled from."""

    _dist: SupportsSampling
    _backend_sample_info: SampleCompatibility

    @property
    def _sample(self) -> Callable[[SupportsRNG, ArrayLike], ArrayLike]:
        """Method for drawing samples from the backend object."""
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
        backend_sample_shape_arg: str = "sample_shape",
    ) -> None:
        """
        Create a new Distribution.

        Args:
            backend_distribution (SupportsSampling): Backend object that supports
                drawing random samples.
            backend_sample_method (str): Name of the method belonging to
                ``backend_distribution`` that supports drawing samples.
            backend_sample_key_arg (str): Keyword argument of the
                ``backend_sample_method`` to provide the random number seed or key
                object to.
            backend_sample_shape_arg (str): Keyword argument of the
                ``backend_sample_method`` to provide the desired sample shape to.

        """
        self._dist = backend_distribution

        # Setup sampling calls, and perform one-time check for compatibility
        self._backend_sample_info = SampleCompatibility(
            method=backend_sample_method,
            rng_key=backend_sample_key_arg,
            sample_shape=backend_sample_shape_arg,
        )
        self._backend_sample_info.validate_compatible(backend_distribution)

    def sample(self, rng_key: SupportsRNG, sample_shape: ArrayLike = ()) -> ArrayLike:
        """
        Draw samples from the distribution.

        Args:
            rng_key (SupportsRNG): Key or seed object to generate random samples.
            sample_shape (ArrayLike): Number of samples (and shape) to draw.

        Returns:
            ArrayLike: Randomly-drawn samples from the distribution.

        """
        return self._sample(rng_key, sample_shape)
