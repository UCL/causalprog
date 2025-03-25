# First steps: simple working example

As a starting point for the package, we will focus on implementing the required functionality to solve the simplified problem detailed below.
During the description, we will highlight areas that will need to be designed with further generality in mind.
Once the simplified example is working and tested, we will begin relaxing the assumptions that reduced us to the simplified problem - both mathematically and programmatically - and gradually expand the scope of the package outwards, from this basis.

## Problem Statement

We take two random variables $X$ and $Y$ with the following structural equations;

$$ f_X = \mathcal{N}(\mu_X, \nu_X^2), \quad f_Y = \mathcal{N}(X, \nu_Y^2), $$

where we will assume, for the time being, that $\nu_X, \nu_Y \in [0,\infty)$ are fixed values, and $\mu_X$ is a parameter in our causal model.
This means that the causal model, $\Theta = \{\mu_X\}$ has only one parameter for the time being.

Our causal estimand of interest will simply be the expectation of our "outcome" variable $Y$;

$$ \sigma = \mathbb{E}[Y]. $$

Our constraints will be the observed mean of $X$, $\phi = \mathbb{E}[X]$, and we will take our distance function to be $\mathrm{dist}(\phi, \psi) = \vert \phi - \psi \vert$ (essentially a 1-dimensional $L^2$-norm).
For a given tolerance $\epsilon$, we thus have the following problem to solve;

$$ \max_{\mu_X} / \min_{\mu_X} \mathbb{E}[Y], \quad\text{subject to}\quad \vert \phi_{\mathrm{data}} - \mathbb{E}[X] \vert \leq \epsilon. $$

By the structural equations, we can infer that $\mathbb{E}[Y] = \mathbb{E}[X] = \mu_X$, thus reaching

$$ \max_{\mu_X} / \min_{\mu_X} \mu_X, \quad\text{subject to}\quad \vert \phi_{\mathrm{data}} - \mu_X \vert \leq \epsilon. $$

The solution to this problem is $\mu_X = \phi_\mathrm{data} \pm \epsilon$ (the positive solution corresponding to the maximisation).

The purpose of solving this simple problem is that it will force us (as developers) to answer a number of structural questions about the package.

## Generalising

Once we have a working implementation of the above problem, we will begin generalising the problem above and expanding the functionality of the package to match.
Some immediate ideas for generalisations are as follows:

- Generalise to multiple parameters. The example above could - in particular - have $\nu_X$ and $\nu_Y$ be parameters to the model too. This will require us to ensure that whatever general classes we have to represent the structural equations are broad enough to cope with arbitrarily-shaped parameters.
  - A related task is to generalise the structural equations to other (non-normal) distributions first, and then on to more complex "distributions" like feed-forward networks.
- Generalise to multiple constraints. Immediate possibilities would be imposing constraints on the variance of $X$ and/or $Y$, particularly if we have already made $\nu_X$ and/or $\nu_Y$ parameters themselves (rather than fixed values).
- Generalise to arbitrary distance functions. This is likely some low-hanging fruit at first, but has the potential to be quite complex if we later want to use functionality like "auto-diff" to speed up solving the optimisation problem.
- Generalising to arbitrary causal estimates.
  - Though not particularly general, having a method or function that applies the $\mathrm{do}$ operator would be of interest beyond just use in a causal estimand. Though for the time being, we might only want it to be applicable to root nodes.
