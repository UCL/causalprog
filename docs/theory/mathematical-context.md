# Causal Problems and `causalprog`

TL;DR, `causalprog` solves

$$ \max_{\Theta} / \min_{\Theta} \sigma(\Theta), \quad \text{subject to } \quad \vert\vert \phi_{\mathrm{data}, k} - \phi_k(\Theta) \vert\vert\leq \epsilon, \quad \forall k, $$

given

- a [model parameters](./glossary.md#model-parameter) for a [causal model](./glossary.md#causal-model) $\Theta$,
- a [causal estimand](./glossary.md#causal-estimand) $\sigma$,

and [constraint functions](./glossary.md#constraint-function) $\phi = (\phi_k)_k$, where;

- $\phi_\mathrm{data}$ is empirically observed values of $\phi$,
- $\phi = (\phi_k)_k$ is the analytical estimate of $\phi$ from the causal model, given $\Theta$,
- $\vert\vert\cdot\vert\vert$ is a non-negative valued distance function (such as a suitable norm),
- $\epsilon = (\epsilon_k)_k$ is the [tolerance in the observed data](./glossary.md#tolerance-of-a-constraint).

The solution to a causal problem is;

- the maximum / minimum value of the causal estimand $\sigma$,
- and the corresponding set of model parameter values $\Theta$ that allows $\sigma$ to attain this extrema.

The causal estimand $\sigma$ is typically a quantity of interest, derived from our model, that we are unable to empirically observe (or is unfeasible for us to observe).
The constraint functions $\phi_k$ represent quantities derived from our model that we can (and have) observed, and are used along with the tolerance values $\epsilon$ to limit the class of admissible models to those which it was feasible for us to be observing empirically.
Solving the resulting causal problem thus provides us with best / worst case estimates for $\sigma$, given what we know to be true about the real world.

`causalprog` provides utility for setting up causal problems [using DAGs](./glossary.md#abbreviations), which can then be solved via your favourite stochastic optimiser and minimisation algorithm.
For example, one could seek the saddle points of the augmented lagrangian

$$ \mathcal{L}(\Theta, \lambda) := \sigma(\Theta) - \lambda \left( \vert\vert \phi_\mathrm{data} - \phi(\Theta) \vert\vert - \epsilon\right), $$

The package also provides some basic wrappers for these solvers, for the most common techniques / algorithms that are used to solve the optimisation problems that are encountered.

## Causal Estimands

For the time being, `causalprog` focuses on casual estimands that are predominantly integrals of some type.
In particular, the focus is on causal estimands that are the expectations (or possibly higher moments) of one of the random variables $X_k$ given some other conditions.
As such, computing the value of a causal estimand will be done largely through Monte Carlo sampling to approximate these integrands.
Since no assumption is made on the dimensionality of our random variables (and thus domains of the integrals), some of these integrals may require a large number of samples before giving a suitable approximation to the true value.

### The $\mathrm{do}$ Operator

One particular estimand of interest is the effect of _do_-ing something, described by the $\mathrm{do}$ operator.
The expected value of $X_k$ given that we "do" $X_l = x^*$ is written as $\mathbb{E}[ X_k \vert \mathrm{do}(X_l = x^*) ]$.
In general, this is different from $\mathbb{E}[ X_k \vert X_l = x^* ]$.

However, the $\mathrm{do}$ operator has a relatively simple-to-explain effect on $\Theta$; essentially replace (the function) $f_{X_l}$ with the constant $x^*$ (or the appropriate mathematical object it defines).

In the simple case where we have a random variable $Y$ which depends on $X$ (which we can control or fix) and $U$ (which we cannot control), we have that

$$\mathbb{E}[ Y \vert \mathrm{do}(X = x^*) ] = \int f_{Y}(x^*, u) \ \mathrm{d}u \\ \approx \frac{1}{M} \sum_{i=1}^M f_Y(x^*, u^{(i)}),$$

with the approximation following from a Monte Carlo estimation of the integrand using samples $u^{(i)}$ drawn from $U$.
