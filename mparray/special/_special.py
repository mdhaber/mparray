import sys as sys
import numpy as np
from mpmath import mp
from mparray import log, exp, real, asarray, vectorize
from scipy import special

# add imported names to `imports` to avoid altering their documentation
imports = {'sys', 'np', 'mp', 'log', 'exp', 'vectorize',
           'real', 'asarray', 'special', 'imports'}

expm1 = vectorize(mp.expm1)
log1p = vectorize(mp.log1p)
factorial2 = vectorize(mp.fac2)
psi = vectorize(mp.digamma)
digamma = psi
ndtr = vectorize(mp.ncdf)
gamma = vectorize(mp.gamma)
gammaln = vectorize(mp.loggamma)
erf = vectorize(mp.erf)
erfc = vectorize(mp.erfc)
zeta = vectorize(mp.zeta)
poch = vectorize(mp.rf)
binom = vectorize(mp.binomial)
comb = binom
lambertw = vectorize(mp.lambertw)
powm1 = vectorize(mp.powm1)
hyp1f1 = vectorize(mp.hyp1f1)
hyp2f1 = vectorize(mp.hyp2f1)
iv = vectorize(mp.besseli)
kv = vectorize(mp.besselk)


@vectorize
def gammainc(a, x):
    return mp.gammainc(a, a=0, b=x, regularized=True)


@vectorize
def gammaincc(a, x):
    return mp.gammainc(a, a=x, b=mp.inf, regularized=True)


@vectorize
def ndtri(x):
    if x == 0:
        return -mp.inf
    if x == 1:
        return mp.inf
    if x < 0 or x > 1:
        return mp.nan

    extra_dps = int(mp.ceil(-mp.log10(x)))
    with mp.workdps(mp.dps + extra_dps):
        return mp.sqrt(2) * mp.erfinv(2 * x - mp.one)


@vectorize
def log_ndtr(x):
    if x <= 0:
        return mp.log(mp.ncdf(x))
    else:
        complement = mp.ncdf(-x)
        return mp.log1p(-complement)


@vectorize
def betaln(x, y):
    return mp.log(mp.beta(x, y))


@vectorize
def betainc(a, b, x):
    if x < 0 or x > 1:
        # The mpmath betainc implementation is defined on entire real line,
        # (and the complex plane). We want to match scipy.special.betainc.
        return mp.nan
    return mp.betainc(a, b, 0, x, regularized=True)


@vectorize
def fdtr(dn, dd, x):
    return mp.betainc(dn/2, dd/2, 0, x*dn/(dd + x*dn), regularized=True)


@vectorize
def fdtrc(dn, dd, x):
    return mp.betainc(dn/2, dd/2, x*dn/(dd + x*dn), 1, regularized=True)


@vectorize
def xlogy(x, y):  # needs accuracy review
    return x*mp.log(y)


@vectorize
def xlog1py(x, y):  # needs accuracy review
    return x*mp.log1p(y)


@vectorize
def cosm1(x):
    # second term in cosine series is x**2/2
    extra_dps = 2*int(mp.ceil(-mp.log10(x))) + 1
    mp.dps += extra_dps
    res = mp.cos(x) - mp.one
    mp.dps -= extra_dps
    return res


@vectorize
def logit(x):  # needs accuracy review
    res = mp.log(x) - mp.log1p(-x)
    return res


@vectorize
def expit(x):  # needs accuracy review
    return mp.exp(x - mp.log1p(mp.exp(x)))


@vectorize
def boxcox(x, lmbda):  # needs accuracy review
    """
    y = (x**lmbda - 1) / lmbda  if lmbda != 0
        log(x)                  if lmbda == 0
    """
    if lmbda != 0:
        return mp.powm1(x, lmbda) / lmbda
    else:
        return mp.lmbda(x)


def boxcox1p(x, lmbda):  # needs accuracy review
    return boxcox(mp.one + x, lmbda)


def logsumexp(a, axis=None, b=None):
    # As far as I know, logsumexp is to avoid overflow, not to improve precision.
    # mpmath doesn't overflow, so naive implementation should be OK.
    return log((b*exp(a)).sum(axis=axis))


def ive(v, z):
    return asarray(iv(v, z) * exp(-abs(real(z))))


def i0e(x):
    return asarray(ive(0, x))


def i1e(x):
    return asarray(ive(1, x))


def kve(v, z):
    return asarray(kv(v, z) * exp(z))


def k0e(x):
    return asarray(kve(0, x))


def k1e(x):
    return asarray(kve(1, x))


def chdtr(v, x):
    return asarray(gammainc(v / 2, x / 2))


def chdtrc(v, x):
    return asarray(gammaincc(v / 2, x / 2))


def stdtr(df, t):
    x = df / (t**2 + df)
    p = betainc(df/2, mp.one/2, x)/2
    return np.where(t < 0, p, mp.one - p)


# others to be added
# gammaincinv
# gammainccinv
# chdtri
# chndtr
# chndtrix
# stdtrit
# ndtri_exp
# tklmbda
# inv_boxcox
# inv_boxcox1p
# kolmogorov, smirnov
# erfcinv
# erfinv


# generate rough documentation
function_names = list(sys.modules[__name__].__dict__.keys())
for key in function_names:
    if key in imports or '_' in key:
        continue
    sys.modules[__name__].__dict__[key].__doc__ = getattr(special, key).__doc__
