# Ricardo's Graph: Reduction to Linear Problem with Quadratic Constraint

This document assumes (and uses) Ricardo's graph setup, it's notation, etc.
We aim to setup a problem that we can analytically solve, to use as an integration test for our package.

To that end, we will set the following:

<!-- prettier-ignore -->
\begin{align*}
\sigma_{czl} = \frac{1}{2}\mathbb{I}_{d_z}, \quad
\mathcal{D}_{eval} = \left\{ (\tilde{x}, \tilde{z}, \tilde{l} ) \right\}, \quad
g(x, z, l) = -x \mathbb{I}_{d_z}, \quad
\hat{r}(x, z, l) = 0, \quad
f_Y(u_y, x, l) = \frac{\theta_Y}{l}(u_y - x)^2,
\end{align*}

where $\mathbb{I}_{d_z} = \frac{1}{\sqrt{d_z}}(1, 1, 1, ...)^{\top}\in\mathbb{R}^{d_Z}$ and $(\tilde{x}, \tilde{z}, \tilde{l} )$ is some chosen evaluation point (of which, only $\tilde{x}$ will turn out to be relevant).

Note that this effectively forces us to pick constant functions for $f_r$ (constant value $\infty$) and $f_m$ (constant value 0), and thus also gives us $\theta_m$ and $\theta_r$ independent problems.
As we will shortly see, the choice for $f_{\pi}$ will also be irrelevant due to the nature of the problem we are getting up, so for argument's sake it can just map to the constant 1-vector (and the problem is independent of $\theta_{\pi}$ too).

Let us also define $\alpha(x) = \frac{3}{4}(1 + 3x^2) > 0$, which is a constant with respect to the parameters of the problem $\theta$.
We can also immediately deduce that

<!-- prettier-ignore -->
\begin{align*}
m_y = \sigma_{czl}^{\top}g(x, z, l) = -\frac{x}{2}, \quad
v_y = \frac{3}{4}.
\end{align*}

We can then make the following simplifications to the forms of the regression function, learn initialiser, and causal responses:

<!-- prettier-ignore -->
\begin{align*}
r(x, z, l)
& = \int f_Y(u_y, x, l) \sum_{c}\pi_{ul}(c)p_N(u_y; m_y, v_y) \ \mathrm{d}u_y
= \int\frac{\theta_Y}{l}(u_y - x)^2 \sum_{c}\pi_{ul}(c)p_N(u_y; -\frac{x}{2}, \frac{3}{4}) \ \mathrm{d} u_y                \\
& = \int\frac{\theta_Y}{l}(u_y - x)^2 p_N(u_y; -\frac{x}{2}, \frac{3}{4}) \ \mathrm{d} u_y
= \frac{\theta_Y}{l}\mathbb{E}\left[(U - x)^2 \ | \ U\sim\mathcal{N}\left(-\frac{x}{2}, \frac{3}{4}\right)\right]          \\
& = \frac{\theta_Y}{l}\left(\frac{3}{4} + \left(\frac{-x}{2}\right)^2 - 2x\left(-\frac{x}{2}\right) + x^2\right)
= \frac{\theta_Y \alpha(x)}{l},                                                                                            \\
B(\theta) & = \frac{1}{n_{eval}}\sum_{\mathcal{D}_{eval}}\left(\hat{r}_i - r(x^{(i)}, z^{(i)}, l^{(i)}) \right)^2
= \frac{\theta_Y^2 \alpha(\tilde{x})^2}{l^2},                                                                              \\
d(x, l)   & = \int f_Y(u_y, x, l)p_N(u_y; 0, 1) \ \mathrm{d}u_y
= \frac{\theta_Y}{l}(1 + x^2).
\end{align*}

This means that we have $\theta^{\star} = \left\{ \theta_Y = 0 \right\}$, since $B(\theta^{\star}) = 0$.
Note that we have used the fact that $\sum_{c}\pi_{ul}(c) = 1$, since we now have that everything else in the integrand is $c$-independent.

Therefore, given $\epsilon = \delta^2$, our problem

<!-- prettier-ignore -->
\begin{align*}
\mathrm{min} / \mathrm{max}_{\theta} d(x, l; \theta)
  & \quad\text{ subject to }\quad
B(\theta) \leq B(\theta^{\star}) + \epsilon,
\end{align*}

reduces to

<!-- prettier-ignore -->
\begin{align*}
\mathrm{min} / \mathrm{max}_{\theta} \frac{\theta_Y}{l}(1 + x^2)
  & \quad\text{ subject to }\quad
\frac{\theta_Y^2 \alpha(\tilde{x})^2}{l^2} \leq \delta^2.
\end{align*}

Furthermore, the constraint is now a simple quadratic in $\theta_Y$ which we can solve for, giving

<!-- prettier-ignore -->
\begin{align*}
-\frac{\delta l}{\alpha(\tilde{x})} \leq \theta_Y \leq \frac{\delta l}{\alpha(\tilde{x})},
\end{align*}

which then lets us immediately read off the solution to the maximisation and minimisation problem:

<!-- prettier-ignore -->
\begin{align*}
\text{max: attained at } \theta_Y = \frac{\delta l}{\alpha(\tilde{x})},
  & \quad\text{with objective value } \quad \frac{\delta(1 + x^2)}{\alpha(\tilde{x})},  \\
\text{min: attained at } \theta_Y = -\frac{\delta l}{\alpha(\tilde{x})}
  & \quad\text{with objective value } \quad -\frac{\delta(1 + x^2)}{\alpha(\tilde{x})}. \\
\end{align*}

Ergo, for given choices of $x$, $\tilde{x}$, and $\delta$, we now have an analytic solution that we can compare against.
