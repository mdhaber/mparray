import numpy as np
from mpmath import mp
from mparray import log, exp, real, mparray


def double_working_precision(f):
    def wrapped(*args, **kwargs):
        mp.dps *= 2
        res = f(*args, **kwargs)
        mp.dps //= 2
        return res

    return wrapped


def working_precision_for_complement(f):
    def wrapped(x, *args, **kwargs):
        extra_dps = int(mp.ceil(-mp.log10(x)))
        mp.dps += extra_dps
        res = f(x, *args, **kwargs)
        mp.dps -= extra_dps
        return res

    return wrapped


# def complement(f, sign=1):
#     def wrapped(*args, **kwargs):
#         res = f(*args, **kwargs)
#         extra_dps = int(mp.ceil(-mp.log10(res)))
#         mp.dps += extra_dps
#         res = mp.one - res
#         mp.dps -= extra_dps
#         return res * sign
#
#     return wrapped


expm1 = np.vectorize(mp.expm1)
log1p = np.vectorize(mp.log1p)
factorial2 = np.vectorize(mp.fac2)
psi = np.vectorize(mp.digamma)
digamma = psi
ndtr = np.vectorize(mp.ncdf)
gammaln = np.vectorize(mp.loggamma)
erf = np.vectorize(mp.erf)
erfc = np.vectorize(mp.erfc)
zeta = np.vectorize(mp.zeta)
poch = np.vectorize(mp.rf)
binom = np.vectorize(mp.binomial)
comb = binom
lambertw = np.vectorize(mp.lambertw)
powm1 = np.vectorize(mp.powm1)
hyp1f1 = np.vectorize(mp.hyp1f1)
hyp2f1 = np.vectorize(mp.hyp2f1)
iv = np.vectorize(mp.besseli)
kv = np.vectorize(mp.besselk)


@np.vectorize
def gammainc(x, a):
    return mp.gammainc(x, a=0, b=a, regularized=True)


@np.vectorize
def gammaincc(x, a):
    return mp.gammainc(x, a=a, b=mp.inf, regularized=True)


@np.vectorize
def ndtri(x):
    extra_dps = int(mp.ceil(-mp.log10(x)))
    mp.dps += extra_dps
    res = mp.sqrt(2) * mp.erfinv(2 * x - mp.one)
    mp.dps -= extra_dps
    return res


@np.vectorize
def log_ndtr(x):
    if x <= 0:
        return mp.log(mp.ncdf(x))
    else:
        complement = mp.ncdf(-x)
        return mp.log1p(-complement)


@np.vectorize
def betaln(x, y):
    return mp.log(mp.beta(x, y))


@np.vectorize
def betainc(a, b, x):
    return mp.betainc(a, b, 0, x, regularized=True)


@np.vectorize
def fdtr(dn, dd, x):
    return mp.betainc(dn/2, dd/2, 0, x*dn/(dd + x*dn), regularized=True)


@np.vectorize
def fdtrc(dn, dd, x):
    return mp.betainc(dn/2, dd/2, x*dn/(dd + x*dn), 1, regularized=True)


@np.vectorize
def xlogy(x, y):
    return x*mp.log(y)


@np.vectorize
def xlog1py(x, y):
    return x*log1p(y)


@np.vectorize
def cosm1(x):
    # second term in cosine series is x**2/2
    extra_dps = 2*int(mp.ceil(-mp.log10(x))) + 1
    mp.dps += extra_dps
    res = mp.cos(x) - mp.one
    mp.dps -= extra_dps
    return res


@np.vectorize
def logit(x):
    res = mp.log(x) - mp.log1p(-x)  # needs precision near x=0.5
    return res


@np.vectorize
def expit(x):
    return mp.exp(x - mp.log1p(mp.exp(x)))  # needs extra precision


@np.vectorize
def boxcox(x, lmbda):  # precision check
    """
    y = (x**lmbda - 1) / lmbda  if lmbda != 0
        log(x)                  if lmbda == 0
    """
    if lmbda != 0:
        return mp.powm1(x, lmbda) / lmbda
    else:
        return mp.lmbda(x)


def boxcox1p(x, lmbda):  # precision check
    return boxcox(1 + x, lmbda)


def logsumexp(a, axis=None, b=None):
    # As far as I know, logsumexp is to avoid overflow, not to improve precision.
    # mpmath doesn't overflow, so naive implementation should be OK.
    return log((b*exp(a)).sum(axis=axis))


def ive(v, z):
    return mparray(iv(v, z) * exp(-abs(real(z))))


def i0e(x):
    return mparray(ive(0, x))


def i1e(x):
    return mparray(ive(1, x))


def kve(v, z):
    return mparray(kv(v, z) * exp(z))


def k0e(x):
    return mparray(kve(0, x))


def k1e(x):
    return mparray(kve(1, x))


def chdtr(v, x):
    return mparray(gammainc(v / 2, x / 2))


def chdtrc(v, x):
    return mparray(gammaincc(v / 2, x / 2))


# others to be added
# gammaincinv
# gammainccinv
# chdtri
# chndtr
# chndtrix
# stdtr
# stdtrit
# ndtri_exp
# tklmbda
# inv_boxcox
# inv_boxcox1p
# kolmogorov, smirnov
# erfcinv
# erfinv
