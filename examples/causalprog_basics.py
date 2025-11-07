r"""
causalprog Basics
=================

TODO: Copied some sample text from notebook. Need to re-write once the notebook changes
are finalised.

In this example we will setup and solve a simple causal problem, introducing core
`causalprog` features along the way.
We will learn how to define a causal model in `causalprog` via a Directed Acyclic Graph
(DAG), specify a causal estimand and data constraints, and numerically estimate bounds
on the causal estimand.

## The Problem

In this example we will examine a simple causal problem that only involves two model
parameters $\mu_X$ and $\nu_Y$; and two random variables (RVs)

\begin{aligned}
X \sim \mathcal{N}(\mu_{x}, 1),
&\quad
Y \mid X \sim \mathcal{N}(X, \nu_{y}).
\end{aligned}

We wish to calculate bounds for the causal estimand

$$ \sigma(\mu_{x}, \nu_{y}) = \mathbb{E}[Y]$$

given observed data (constraints)

$$ \phi(\mu_{x}, \nu_{y}) = \mathbb{E}[X]$$

and tolerance in the observed data $\epsilon$.

Therefore, we are looking to solve the following minimisation problem:

\begin{align}
\mathrm{max}/\mathrm{min}_{\mu_{x}, \nu_{y}} \mu_{x}, \quad
\text{subject to } \vert \mu_{x} - \phi_{obs} \vert \leq \epsilon.
\end{align}

The solution to this is $\mu_{x}^{*} = \phi_{obs} \pm \epsilon$.
The value of $\nu_{y}$ can be any positive value, since in this setup both $\phi$ and
$\sigma$ are independent of it.
The corresponding Lagrangians are:

\begin{align}
\mathcal{L}_{\min}(\mu_X, \nu_Y, \lambda) =  \mu_X
+ \lambda(|\mu_{x} - \phi_{obs}| - \epsilon) \qquad \lambda \geq 0
\mathcal{L}_{\max}(\mu_{x}, \nu_{y}, \lambda) = - \mu_{x}
+ \lambda(|\mu_{x} - \phi_{obs}| - \epsilon) \qquad \lambda \geq 0
\end{align}

With KKT (primal-dual) solutions
$(\mu_{x}^*, \nu_{y}, \lambda^*) = (\phi_{obs} \pm \epsilon, \nu_{y}, 1)$
In this notebook, with assistance from causalprog, we will attempt to find this solution
using the naive approach of minimising  $\| \nabla \mathcal{L} \|_2^2$.
"""

# %%
# TODO: First image generated is the thumbnail - should create an image of the DAG
# (if only graph.draw...?)
