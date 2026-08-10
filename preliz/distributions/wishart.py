import numpy as np
from pytensor_distributions import wishart as ptd_wishart

from preliz.distributions.continuous_multivariate import Continuous
from preliz.internal.distribution_helper import all_not_none, pytensor_jit, pytensor_rng_jit


class Wishart(Continuous):
    r"""
    Wishart Distribution.

    The pdf of this distribution is

    .. math::

        f(X \mid \nu, V) =
            \frac{|X|^{(\nu - p - 1)/2} \exp\left(-\frac{1}{2}\text{tr}(V^{-1}X)\right)}
            {2^{\nu p/2} |V|^{\nu/2} \Gamma_p\left(\frac{\nu}{2}\right)}

    ========  ==============================================================
    Support   :math:`X` a :math:`p \times p` positive semi-definite matrix
    Mean      :math:`\nu V`
    Variance  :math:`\nu (V_{ij}^2 + V_{ii}V_{jj})`
    ========  ==============================================================

    Parameters
    ----------
    nu : float
        Degrees of freedom, :math:`\nu > p - 1`.
    V : array_like
        Scale matrix, :math:`p \times p` positive definite.
    """

    def __init__(self, nu=None, V=None):
        super().__init__()
        self.support = "positive_definite"
        self._parametrization(nu, V)

    def _parametrization(self, nu=None, V=None):
        self.nu = nu
        self.V = V
        self.param_names = ("nu", "V")
        self.params_support = ((0, np.inf), "positive_definite")
        self.support = ("positive_definite", )
        self.params = (self.nu, self.V)

        if all_not_none(nu, V):
            self._update(nu, V)

    def _update(self, nu, V):
        self.nu = np.float64(nu)
        self.V = np.asarray(V, dtype=np.float64)
        self.params = (self.nu, self.V)
        self.support = (-np.inf, np.inf)
        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.nu, self.V)

    def logpdf(self, x):
        return ptd_logpdf(x, self.nu, self.V)

    def entropy(self):
        return ptd_entropy(self.nu, self.V)

    def mean(self):
        return ptd_mean(self.nu, self.V)

    def mode(self):
        return ptd_mode(self.nu, self.V)

    def var(self):
        return ptd_var(self.nu, self.V)

    def std(self):
        return ptd_std(self.nu, self.V)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.nu, self.V, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        diag_mean = np.diagonal(mean, axis1=-2, axis2=-1) if mean.ndim > 1 else mean
        nu_estimates = 2 * (diag_mean / sigma) ** 2
        nu = np.mean(nu_estimates)
        V = mean / nu
        print(V, nu)
        self._update(nu, V)

    def _fit_mle(self, sample):
        raise NotImplementedError


@pytensor_jit(constants=("V", "nu", "x"))
def ptd_pdf(x, nu, V):
    return ptd_wishart.pdf(x, nu, V)

@pytensor_jit(constants=("V", "nu", "x"))
def ptd_logpdf(x, nu, V):
    return ptd_wishart.logpdf(x, nu, V)

@pytensor_jit(constants=("V", "nu"))
def ptd_entropy(nu, V):
    return ptd_wishart.entropy(nu, V)

@pytensor_jit(constants=("V", "nu"))
def ptd_mean(nu, V):
    return ptd_wishart.mean(nu, V)

@pytensor_jit(constants=("V", "nu"))
def ptd_mode(nu, V):
    return ptd_wishart.mode(nu, V)

@pytensor_jit(constants=("V", "nu"))
def ptd_var(nu, V):
    return ptd_wishart.var(nu, V)

@pytensor_jit(constants=("V", "nu"))
def ptd_std(nu, V):
    return ptd_wishart.std(nu, V)

@pytensor_rng_jit(constants=("V", "nu"))
def ptd_rvs(nu, V, size, rng):
    return ptd_wishart.rvs(nu, V, size=size, random_state=rng)
