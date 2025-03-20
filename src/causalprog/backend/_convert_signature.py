"""Convert a function signature to a different signature."""

import inspect
from collections.abc import Callable
from inspect import Parameter, Signature
from typing import Any

from ._typing import ParamKind, ParamNameMap, ReturnType, StaticValues

_VARLENGTH_PARAM_TYPES = (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)


def _check_variable_length_params(
    sig: Signature,
) -> dict[ParamKind, str | None]:
    """
    Return the names of variable-length parameters in a signature.

    There are two types of variable-length parameters; positional (VAR_POSITIONAL) which
    are typically denoted ``*args`` or ``*vargs``, and keyword (VAR_KEYWORD) which are
    typically denoted ``**kwargs``.

    ``Signature`` objects can contain more than one variable-length parameter of each
    kind, despite the fact that in practice such a signature cannot exist and be valid
    Python syntax. This function checks for such cases, and raises an appropriate error,
    should they arise. Otherwise, it simply identifies the parameters in ``sig`` which
    correspond to these two variable-length parameter kinds.

    Args:
        sig (Signature): Function signature to check for variable-length parameters.

    Returns:
        dict[ParamKind, str | None]: Mapping of variable-length parameter
            kinds to the corresponding parameter name in ``sig``, or to ``None`` if no
            parameter of that type exists in the signature.

    """
    named_args: dict[ParamKind, str | None] = {
        kind: None for kind in _VARLENGTH_PARAM_TYPES
    }
    for kind in _VARLENGTH_PARAM_TYPES:
        possible_parameters = [
            p_name for p_name, p in sig.parameters.items() if p.kind == kind
        ]
        if len(possible_parameters) > 1:
            msg = f"New signature takes more than 1 {kind} argument."
            raise ValueError(msg)
        if len(possible_parameters) > 0:
            named_args[kind] = possible_parameters[0]
    return named_args


def _signature_can_be_cast(
    signature_to_convert: Signature,
    new_signature: Signature,
    old_to_new_names: ParamNameMap,
    give_static_value: StaticValues,
) -> tuple[ParamNameMap, StaticValues]:
    """
    Prepare a signature for conversion to another signature.

    This is a helper that handles the validation detailed in ``convert_signature``.
    See the docstring of ``convert_signature`` for more details.

    Args:
        signature_to_convert (Signature): Function signature that will be cast to
            ``new_signature``.
        new_signature (Signature): See the homonymous argument to ``convert_signature``.
        old_to_new_names (ParamNameMap): See the homonymous argument to
            ``convert_signature``.
        give_static_value (StaticValues): See the homonymous argument to
            ``convert_signature``.

    Raises:
        ValueError: If the two signatures cannot be cast, even given
            the additional information.

    Returns:
        ParamNameMap: Mapping of parameter names in the ``signature_to_convert`` to
            parameter names in ``new_signature``. Implicit mappings as per behaviour of
            ``convert_signature`` are explicitly included in the returned mapping.
        StaticValues: Mapping of parameter names in the ``signature_to_convert`` to
            static values to assign to these parameters, indicating omission from the
            ``new_signature``. Implicit adoption of static values as per behaviour of
            ``convert_signature`` are explicitly included in the returned mapping.

    See Also:
        convert_signature: Function for which setup is being performed.

    """
    _check_variable_length_params(signature_to_convert)
    new_varlength_params = _check_variable_length_params(new_signature)

    old_to_new_names = dict(old_to_new_names)
    give_static_value = dict(give_static_value)

    new_parameters_accounted_for = set()

    # Check mapping of parameters in old signature to new signature
    for p_name, param in signature_to_convert.parameters.items():
        is_explicitly_mapped = p_name in old_to_new_names
        name_is_unchanged = (
            p_name not in old_to_new_names
            and p_name not in old_to_new_names.values()
            and p_name in new_signature.parameters
        )
        is_given_static = p_name in give_static_value
        can_take_default = param.default is not param.empty
        is_varlength_param = param.kind in _VARLENGTH_PARAM_TYPES
        mapped_to = None

        if is_explicitly_mapped:
            # This parameter is explicitly mapped to another parameter
            mapped_to = old_to_new_names[p_name]
        elif name_is_unchanged:
            # Parameter is inferred not to change name, having been omitted from the
            # explicit mapping.
            mapped_to = p_name
            old_to_new_names[p_name] = mapped_to
        elif (
            is_varlength_param
            and new_varlength_params[param.kind] is not None
            and str(new_varlength_params[param.kind]) not in old_to_new_names.values()
        ):
            # Automatically map VAR_* parameters to their counterpart, if possible.
            mapped_to = str(new_varlength_params[param.kind])
            old_to_new_names[p_name] = mapped_to
        elif is_given_static:
            # This parameter is given a static value to use.
            continue
        elif can_take_default:
            # This parameter has a default value in the old signature.
            # Since it is not explicitly mapped to another parameter, nor given an
            # explicit static value, infer that the default value should be set as the
            # static value.
            give_static_value[p_name] = param.default
        else:
            msg = (
                f"Parameter '{p_name}' has no counterpart in new_signature, "
                "and does not take a static value."
            )
            raise ValueError(msg)

        # Record that any parameter mapped_to in the new_signature is now accounted for,
        # to avoid many -> one mappings.
        if mapped_to:
            if mapped_to in new_parameters_accounted_for:
                msg = f"Parameter '{mapped_to}' is mapped to by multiple parameters."
                raise ValueError(msg)
            # Confirm that variable-length parameters are mapped to variable-length
            # parameters (of the same type).
            if (
                is_varlength_param
                and new_signature.parameters[mapped_to].kind != param.kind
            ):
                msg = (
                    "Variable-length positional/keyword parameters must map to each "
                    f"other ('{p_name}' is type {param.kind}, but '{mapped_to}' is "
                    f"type {new_signature.parameters[mapped_to].kind})."
                )
                raise ValueError(msg)

            new_parameters_accounted_for.add(old_to_new_names[p_name])

    # Confirm all items in new_signature are also accounted for.
    unaccounted_new_parameters = (
        set(new_signature.parameters) - new_parameters_accounted_for
    )
    if unaccounted_new_parameters:
        msg = "Some parameters in new_signature are not used: " + ", ".join(
            unaccounted_new_parameters
        )
        raise ValueError(msg)

    return old_to_new_names, give_static_value


def convert_signature(
    fn: Callable[..., ReturnType],
    new_signature: Signature,
    old_to_new_names: ParamNameMap,
    give_static_value: StaticValues,
) -> Callable[..., ReturnType]:
    """
    Convert the call signature of a function ``fn`` to that of ``new_signature``.

    This function effectively allows ``fn`` to be called with ``new_signature``. It
    returns a new ``Callable`` that uses the ``new_signature``, and returns the result
    of ``fn`` after translating the ``new_signature`` back into that of ``fn`` and
    making an appropriate call.

    Converting signatures into each other is, in general, not possible. However under
    certain assumptions and conventions, it can be done. To that end, the following
    assumptions are made about ``fn`` and ``new_signature``:

    1. All parameters to ``fn`` are either;
        1. mapped to one non-variable-length parameter of ``new_signature``, or
        2. provided with a static value to be used in all calls.
    2. If ``fn`` takes a ``VAR_POSITIONAL`` parameter ``*args``, then either
        1. ``new_signature`` must also take a ``VAR_POSITIONAL`` parameter, and this
        must map to identically to ``*args``,
        2. ``*args`` is provided with a static value to be used in all calls, and
        ``new_signature`` must not take ``VAR_POSITIONAL`` arguments.
    3. If ``fn`` takes a ``VAR_KEYWORD`` parameter ``**kwargs``, then either
        1. ``new_signature`` must also take a ``VAR_KEYWORD`` parameter, and this
        must map to identically to ``**kwargs``,
        2. ``**kwargs`` is provided with a static value to be used in all calls, and
        ``new_signature`` must not take ``VAR_KEYWORD`` arguments.

    Mapping of parameters is done by name, from the signature of ``fn`` to
    ``new_signature``, in the ``old_to_new_names`` argument.

    4. If a parameter does not change name between the two signatures, it can be omitted
    from this mapping and it will be inferred. Note that such a parameter may still
    change kind, or adopt a new default value, in the ``new_signature``.

    Parameters can also be "dropped" from ``fn``'s signature in ``new_signature``, by
    assigning them static values to be used in all cases. Such static values are given
    in the ``give_static_value`` mapping, which maps (names of) parameters of ``fn`` to
    a fixed value to be used for that parameter. This means that these parameters do not
    need to be mapped to a parameter in ``new_signature``.

    5. Parameters that have default values in ``fn``, and which are not mapped to a
    parameter of ``new_signature``, will adopt their default value as a static value.

    Args:
        fn (Callable): Callable object to change the signature of.
        new_signature (inspect.Signature): New signature to give to ``fn``.
        old_to_new_names (dict[str, str]): Maps the names of parameters in ``fn``s
            signature to the corresponding parameter names in ``new_signature``.
        give_static_value (dict[str, Any]): Maps names of parameters of ``fn`` to
            static values that should be assigned to them.

    Raises:
        ValueError: If ``fn``'s signature cannot be cast to ``new_signature``, given the
            information provided.

    Returns:
        Callable: Callable representing ``fn`` with ``new_signature``.

    See Also:
        _check_variable_length_params: Validation of number of variable-length
            parameters.
        _signature_can_be_cast: Validation method used to check signatures can be cast.

    """
    fn_signature = inspect.signature(fn)
    old_to_new_names, give_static_value = _signature_can_be_cast(
        fn_signature, new_signature, old_to_new_names, give_static_value
    )
    new_to_old_names = {value: key for key, value in old_to_new_names.items()}

    fn_varlength_params = _check_variable_length_params(fn_signature)
    fn_vargs_param = fn_varlength_params[Parameter.VAR_POSITIONAL]
    fn_kwargs_param = fn_varlength_params[Parameter.VAR_KEYWORD]

    new_varlength_params = _check_variable_length_params(new_signature)
    new_kwargs_param = new_varlength_params[Parameter.VAR_KEYWORD]

    fn_posix_args = [
        p_name
        for p_name, param in fn_signature.parameters.items()
        if param.kind <= param.POSITIONAL_OR_KEYWORD
    ]

    # If fn's VAR_KEYWORD parameter is dropped from the new_signature,
    # it must have been given a default value to use. We need to expand
    # these values now so that they get passed correctly as keyword arguments.
    if fn_kwargs_param and fn_kwargs_param in give_static_value:
        static_kwargs = give_static_value.pop(fn_kwargs_param)
        give_static_value = dict(give_static_value, **static_kwargs)

    def fn_with_new_signature(*args: tuple, **kwargs: dict[str, Any]) -> ReturnType:
        bound = new_signature.bind(*args, **kwargs)
        bound.apply_defaults()

        all_args_received = bound.arguments
        kwargs_to_pass_on = (
            all_args_received.pop(new_kwargs_param, {}) if new_kwargs_param else {}
        )
        # Maps the name of a parameter to fn to the value that should be supplied,
        # as obtained from the arguments provided to this function.
        # Calling dict with give_static_value FIRST is important, as defaults will get
        # overwritten by any passed arguments!
        fn_kwargs = dict(
            give_static_value,
            **{
                new_to_old_names[key]: value for key, value in all_args_received.items()
            },
            **kwargs_to_pass_on,
        )
        # We can supply all arguments EXCEPT the variable-positional and positional-only
        # arguments as keyword args.
        # Positional-only arguments have to come first, followed by the
        # variable-positional parameters.
        fn_args = [fn_kwargs.pop(p_name) for p_name in fn_posix_args]
        if fn_vargs_param:
            fn_args.extend(fn_kwargs.pop(fn_vargs_param, []))
        # Now we can call fn
        return fn(*fn_args, **fn_kwargs)

    return fn_with_new_signature
