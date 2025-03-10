---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Student's t Distribution

<audio controls> <source src="../../_static/studentt.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

[Univariate](../../gallery_tags.rst#univariate), [Continuous](../../gallery_tags.rst#continuous), [Symmetric](../../gallery_tags.rst#symmetric), [Unbounded](../../gallery_tags.rst#unbounded), [Heavy-tailed](../../gallery_tags.rst#heavy-tailed)

The Student's t distribution, also known as the t-distribution, is a continuous probability distribution that resembles the normal distribution but with heavier tails. It is characterized by its bell-shaped curve, symmetric around the mean, and can defined by three parameters: the degrees of freedom ($\nu$), the location parameter ($\mu$), and the scale parameter ($\sigma$). The smaller the value of ($\nu$), the heavier the tails of the distribution.

It is often used in Bayesian analysis particulary as a robust alternative to the Normal due to the possibility of having heavier tails.

## Key properties and parameters

```{eval-rst}
========  ==========================================
Support   :math:`x \in \mathbb{R}`
Mean      :math:`\mu` for :math:`\nu > 1`, otherwise undefined
Variance  :math:`\frac{\nu}{\nu-2}` for :math:`\nu > 2`,
          :math:`\infty` for :math:`1 < \nu \le 2`, otherwise undefined
========  ==========================================
```

**Parameters:**

- $\nu$ : (float) Degrees of freedom, $\nu > 0$.
- $\mu$ : (float) Location parameter.
- $\sigma$ : (float) Scale parameter, $\sigma > 0$.

**Alternative parametrization**

The Student's t distribution has two alternative parameterizations. In terms of $\nu$, $\mu$, and $\sigma$, or in terms of $\nu$, $\mu$ and $\lambda$.

The link between the 2 alternatives is given by:

$$
\lambda = \frac{1}{\sigma^2}
$$

where $\sigma$ is the standard deviation as $\nu$ increases, and $\lambda$ is the precision as $\nu$ increases.

### Probability Density Function (PDF)

$$
f(x \mid \nu, \mu, \sigma) =  \frac{\Gamma \left(\frac{\nu+1}{2} \right)} {\sqrt{\nu\pi}\Gamma \left(\frac{\nu}{2} \right)} \left(1+\frac{x^2}{\nu} \right)^{-\frac{\nu+1}{2}}
$$

where $\Gamma$ is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function).

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\nu$, $\mu$, and $\sigma$
:sync: nu-mu-sigma
```{jupyter-execute}
:hide-code:

from preliz import StudentT, style
style.use('preliz-doc')
nus = [2., 5., 5.]
mus = [0., 0.,  -4.]
sigmas = [1., 1., 2.]
for nu, mu, sigma in zip(nus, mus, sigmas):
    StudentT(nu, mu, sigma).plot_pdf(support=(-10,6))
```
:::::

:::::{tab-item} Parameters $\nu$, $\mu$, and $\lambda$
:sync: nu-mu-lambda
```{jupyter-execute}
:hide-code:

lambdas = [1., 1., 0.25]
for nu, mu, lam in zip(nus, mus, lambdas):
    StudentT(nu, mu, lam=lam).plot_pdf(support=(-10,6))
```
:::::
::::::

### Cumulative Distribution Function (CDF)

$$
F(y \mid \nu, \mu, \sigma) = 
\begin{cases} 
1 - \frac{1}{2} I_{\frac{\nu}{x^2 + \nu}} \left( \frac{\nu}{2}, \frac{1}{2} \right) & \text{for } x = \frac{y - \mu}{\sigma} \leq 0, \\[0.5em]
\frac{1}{2} I_{\frac{\nu}{x^2 + \nu}} \left( \frac{\nu}{2}, \frac{1}{2} \right) & \text{for } x = \frac{y - \mu}{\sigma} > 0,
\end{cases}
$$

where $I_x(a, b)$ denotes the [regularized incomplete beta function](https://en.wikipedia.org/wiki/Regularized_incomplete_beta_function).

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\nu$, $\mu$, and $\sigma$
:sync: nu-mu-sigma
```{jupyter-execute}
:hide-code:

for nu in nus:
    StudentT(nu, mu, sigma).plot_cdf(support=(-10,6))
```
:::::

:::::{tab-item} Parameters $\nu$, $\mu$, and $\lambda$
:sync: nu-mu-lambda
```{jupyter-execute}
:hide-code:

for nu in nus:
    StudentT(nu, mu, lam=lam).plot_cdf(support=(-10,6))
```
:::::
::::::

```{seealso}
:class: seealso

**Common Alternatives:**

- [Skewed Student's t](skew_studentt.md) - Extends the Student's t-distribution by introducing a skewness parameter, allowing for the modeling of data that is not symmetrically distributed. 
- [Half-Student's t](halfstudentt.md) - Considers only the positive values of the Student's t-distribution. 
- [Normal](normal.md) - When $\nu \to \infty$, the t-distribution converges to the normal distribution.
- [Cauchy](cauchy.md) - The Cauchy distribution is a special case of the Student's t-distribution with $\nu=1$.
```
## References

- Wikipedia. [Student's t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution)