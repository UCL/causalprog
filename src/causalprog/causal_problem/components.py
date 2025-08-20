"""Classes for defining causal estimands and constraints of causal problems."""

from causalprog.causal_problem._base_component import _CPComponent


class CausalEstimand(_CPComponent):
    """
    A Causal Estimand.

    The causal estimand is the function that we want to minimise (and maximise)
    as part of a causal problem. It should be a scalar-valued function of the
    random variables appearing in a graph.
    """


class Constraint(_CPComponent):
    r"""
    A Constraint that forms part of a causal problem.

    Constraints of a causal problem are derived properties of RVs for which we
    have observed data. The causal estimand is minimised (or maximised) subject
    to the predicted values of the constraints being close to their observed
    values in the data.

    Adding a constraint $g(\theta)$ to a causal problem (where $\theta$ are the
    parameters of the causal problem) essentially imposes an additional
    constraint on the minimisation problem;

    $$ g(\theta) - g_{\text{data}} \leq \epsilon, $$

    where $g_{\text{data}}$ is the observed data value for the quantity $g$,
    and $\epsilon$ is some tolerance.
    """

    # TODO: (https://github.com/UCL/causalprog/issues/89)
    # Should explain that Constraint needs more inputs and slightly different
    # interpretation of the `do_with_samples` object.
    # Inputs:
    # - include epsilon as an input (allows constraints to have different tolerances)
    # - `do_with_samples` should just be $g(\theta)$. Then have the instance build the
    #   full constraint that will need to be called in the Lagrangian.
    # - $g$ still needs to be scalar valued? Allow a wrapper function to be applied in
    #   the event $g$ is vector-valued.
    # If we do this, will also need to override __call__...
