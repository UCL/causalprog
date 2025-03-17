"""Convert a function signature to a different signature."""

import inspect
from collections.abc import Callable
from inspect import Signature
from typing import Any

from ._typing import ParamNameMap, ReturnType, StaticValues


def _signature_can_be_cast(
    signature_to_convert: Signature,
    new_signature: Signature,
    param_name_map: ParamNameMap,
    give_static_value: StaticValues,
) -> tuple[str | None, ParamNameMap, StaticValues]:
    """
    Prepare a signature for conversion to another signature.

    In order to map ``signature_to_convert`` to that of ``new_signature``, the following
    assurances are needed:

    - Variable positional parameters in the two signatures are assumed to match (even if
    the name of this parameter changes).
    - All parameters WITHOUT DEFAULT VALUES in ``signature_to_convert`` correspond to a
    parameter in ``new_signature`` (that may or may not have a default value) OR are
    given static values to use, via the ``give_static_value`` argument.
    - If ``new_signature`` takes variable-keyword-argument (``**kwargs``), these
    arguments are expanded to allow for possible matches to parameters of
    ``signature_to_convert``, before passing any remaining parameters after this
    unpacked to the variable-keyword-argument of ``signature_to_convert``.

    Args:
        signature_to_convert (Signature): Function signature that will be cast to
            ``new_signature``.
        new_signature (Signature): See the homonymous argument to ``convert_signature``.
        param_name_map (ParamNameMap): See the homonymous argument to
            ``convert_signature``.
        give_static_values (StaticValues): See the homonymous argument to
            ``convert_signature``.

    Raises:
        ValueError: If the two signatures cannot be cast, even given
        the additional information.

    Returns:
        str | None: The name of the variable-length positional argument parameter in the
            ``signature_to_convert``, or ``None`` if such a parameter does not exist.
        ParamNameMap: Mapping of parameter names in the ``signature_to_convert`` to
            parameter names in ``new_signature``.
        StaticValues: Mapping of parameter names in the ``signature_to_convert`` to
            static values to assign to these parameters, indicating omission from the
            ``new_signature``.

    """
    fn_signature = signature_to_convert

    # Assert matching of variable-number of positional arguments.
    possible_old_varg_params = [
        p_name
        for p_name, param in fn_signature.parameters.items()
        if param.kind is param.VAR_POSITIONAL
    ]
    possible_new_varg_params = [
        p_name
        for p_name, param in new_signature.parameters.items()
        if param.kind is param.VAR_POSITIONAL
    ]
    if possible_old_varg_params:
        # Either both or neither signature must take such a parameter.
        if len(possible_old_varg_params) != len(possible_new_varg_params):
            msg = (
                "Either both signatures, or neither, "
                "must accept a variable number of positional arguments."
            )
            raise ValueError(msg)
        # There can only be one variable-length positional argument in a signature,
        # so it is safe to take the 0th element as we know both lists are non-empty
        # and of equal length.
        old_varg_param = possible_old_varg_params[0]
        new_varg_param = possible_new_varg_params[0]
        # Validate the mapping between these parameters.
        if (
            old_varg_param != new_varg_param
            and param_name_map[new_varg_param] != old_varg_param
        ):
            msg = (
                f"Variable-positional parameter ({old_varg_param}) is not mapped "
                "to another variable-positional parameter."
            )
            raise ValueError(msg)
    else:
        old_varg_param = None

    # Record any static values to give to parameters to fn that don't have a counterpart
    # in new_signature.
    give_static_value = dict(give_static_value)
    # Inherit existing default values for fn's parameters, if new ones are not provided.
    for p_name, param in fn_signature.parameters.items():
        if p_name not in give_static_value and param.default is not param.empty:
            give_static_value[p_name] = param.default

    param_name_map = dict(param_name_map)
    # Confirm that we have sufficient information to cast fn's signature to the
    # new_signature.
    for p_name, param in fn_signature.parameters.items():
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            # Is this parameter mapped to a parameter of a different name in the new
            # signature?
            has_name_change = p_name in param_name_map.values()
            # Is this parameter given a static value?
            has_static_value = p_name in give_static_value
            # Does this parameter match the name of a parameter in new_signature?
            matches_param_in_new = p_name in new_signature.parameters.values()
            if not (has_name_change or has_static_value):
                # This parameter is required and does not change name nor take a static
                # value, so must retain its name in the new signature.
                if matches_param_in_new:
                    # This parameter is not explicitly mapped, so add this association.
                    param_name_map[p_name] = p_name
                else:
                    msg = f"{p_name} is not mapped to a parameter in the new signature!"
                    raise ValueError(msg)

    return old_varg_param, param_name_map, give_static_value


def convert_signature(
    fn: Callable[..., ReturnType],
    new_signature: Signature,
    param_name_map: ParamNameMap,
    give_static_value: StaticValues,
) -> Callable[..., ReturnType]:
    """
    Convert the call signature of a function ``fn`` to that of ``new_signature``.

    Args:
        fn (Callable): Callable object to change the signature of.
        new_signature (inspect.Signature): New signature to give to ``fn``.
        param_name_map (dict[str, str]): Maps the names of parameters in the new
            signature to the corresponding parameter names in ``fn``s signature.
            Parameter names that do not change can be omitted. Note that parameters that
            are to be dropped should be supplied to ``give_static_value`` instead.
        give_static_value (dict[str, Any]): Maps names of parameters of ``fn`` to
            default values that should be assigned to them. This means that not all
            compulsory parameters of ``fn`` have to have a corresponding parameter in
            ``new_signature`` - such parameters will use the value assigned to them in
            ``give_static_value`` if they are lacking a counterpart parameter in
            ``new_signature``. Parameters to ``fn`` that lack a counterpart in
            ``new_signature``, and that already have default values, will inherit them
            here if not provided explicitly.

    Returns:
        Callable: Callable representing ``fn`` with ``new_signature``.

    See Also:
        _signature_can_be_cast: Validation method used to check casting is possible.

    """
    fn_signature = inspect.signature(fn)
    fn_vargs_param, param_name_map, give_static_value = _signature_can_be_cast(
        fn_signature, new_signature, param_name_map, give_static_value
    )

    fn_posix_args = [
        p_name
        for p_name, param in fn_signature.parameters.items()
        if param.kind is param.POSITIONAL_ONLY
    ]
    possible_fn_kwargs_params = [
        p_name
        for p_name, param in new_signature.parameters.items()
        if param.kind is param.VAR_KEYWORD
    ]
    new_kwargs_parameter = (
        possible_fn_kwargs_params[0] if possible_fn_kwargs_params else None
    )

    def fn_with_new_signature(*args: tuple, **kwargs: dict[str, Any]) -> ReturnType:
        bound = new_signature.bind(*args, **kwargs)
        bound.apply_defaults()

        all_args_received = bound.arguments
        kwargs_to_pass_on = (
            all_args_received.pop(new_kwargs_parameter, {})
            if new_kwargs_parameter
            else {}
        )
        # Maps the name of a parameter to fn to the value that should be supplied,
        # as obtained from the arguments provided to this function.
        # Calling dict with give_static_value FIRST is important, as defaults will get
        # overwritten by any passed arguments!
        fn_kwargs = dict(
            give_static_value,
            **{param_name_map[key]: value for key, value in all_args_received.items()},
        )
        # We can supply all arguments EXCEPT the variable-positional and positional-only
        # arguments as keyword args.
        # Positional-only arguments have to come first, followed by the
        # variable-positional parameters.
        fn_args = [fn_kwargs.pop(p_name) for p_name in fn_posix_args]
        if fn_vargs_param:
            fn_args.extend(fn_kwargs.pop(fn_vargs_param, []))
        # Now we can call fn
        return fn(*fn_args, **fn_kwargs, **kwargs_to_pass_on)

    return fn_with_new_signature
