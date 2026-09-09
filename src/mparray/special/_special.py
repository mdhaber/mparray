import sys as sys

from mpmath import mp
from scipy import special

import mparray as xp
from mparray._mparray import _promote as promote
from mparray._mparray import _vectorize as vectorize

# add imported names to `_imports` to avoid altering their documentation and exposing
# as public members of `mparray.special`.
_imports = {'sys', 'np', 'mp', 'xp', 'vectorize',
            'special', 'promote', 'fresnelc', 'fresnels'}

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
erfinv = vectorize(mp.erfinv)
zeta = vectorize(mp.zeta)
poch = vectorize(mp.rf)
binom = vectorize(mp.binomial)
comb = binom
powm1 = vectorize(mp.powm1)
hyp1f1 = vectorize(mp.hyp1f1)
hyp2f1 = vectorize(mp.hyp2f1)
iv = vectorize(mp.besseli)
kv = vectorize(mp.besselk)
lambertw = vectorize(mp.lambertw)
fresnels = vectorize(mp.fresnels)
fresnelc = vectorize(mp.fresnelc)


@vectorize
def erfcx(x):
    return mp.exp(x**2) * mp.erfc(x)


@vectorize
def erfcinv(p):
    if not (p >= 0 and p <= 2):
        return mp.nan

    if p == 0:
        return -mp.inf
    elif p == 2:
        return mp.inf

    def f(t):
        return (mp.erfc(t) - p)/abs(p)

    b = 1
    while not (f(-b) > 0 > f(b)):
        b = b*2
    return mp.findroot(f, (-b, b), solver='illinois', maxsteps=1000)


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
def ndtri_exp(p):
    if not p <= 0:
        return mp.nan

    if p == -mp.inf:
        return -mp.inf
    elif p == 0:
        return mp.inf

    def f(t):
        res = (mp.log(mp.ncdf(t)) if t <= 0 else mp.log1p(-mp.ncdf(-t))) - p
        return res / abs(p) if abs(p) < 1 else res

    b = -1
    while f(b) > 0:
        b = b*2
    return mp.findroot(f, (b, 0), solver='illinois', maxsteps=1000)


@vectorize
def beta(x, y):
    return mp.beta(x, y)


@vectorize
def betaln(x, y):
    return mp.log(mp.beta(x, y))


@vectorize
def betainc(a, b, x):
    return mp.betainc(a, b, 0, x, regularized=True)


@vectorize
def fdtr(dn, dd, x):
    return mp.betainc(dn/2, dd/2, 0, x*dn/(dd + x*dn), regularized=True)


@vectorize
def fdtrc(dn, dd, x):
    return mp.betainc(dn/2, dd/2, x*dn/(dd + x*dn), 1, regularized=True)


@vectorize
def fdtri(dn, dd, p):
    if not ((dn >= 0) and (dd >= 0) and (p >= 0 and p <= 1)):
        return mp.nan

    if p == 0:
        return -mp.inf
    elif p == 1:
        return mp.inf

    def f(x):
        return mp.betainc(dn/2, dd/2, 0, x*dn/(dd + x*dn), regularized=True) - p

    b = 1
    while f(b) < 0:
        b = b*2
    return mp.findroot(f, (0, b), solver='illinois', maxsteps=1000)


def pdtr(k, m):
    return gammaincc(xp.floor(k+1), m)


def pdtrc(k, m):
    return gammainc(xp.floor(k+1), m)


@vectorize
def xlogy(x, y):
    return 0 if (x == 0 and not xp.isnan(y)) else x*mp.log(y)


@vectorize
def xlog1py(x, y):
    return 0 if (x == 0 and not xp.isnan(y)) else x*mp.log1p(y)


@vectorize
def cosm1(x):
    if x == 0:
        # Handle this case separately to avoid blow up in extra_dps calculation.
        return mp.zero
    # second term in cosine series is x**2/2
    # catastrophic cancellation also occurs near nonzero multiples of 2*pi,
    # but doubling precision is enough here. We are being conservative by
    # always at least doubling the precision.
    extra_dps = max(mp.dps, 2*int(mp.ceil(-mp.log10(x))) + 1)
    with mp.workdps(mp.dps + extra_dps):
        return mp.cos(x) - mp.one


@vectorize
def logit(x):  # needs accuracy review
    res = mp.log(x) - mp.log1p(-x)
    return res


def expit(x):  # needs accuracy review
    return xp.exp(log_expit(x))


@vectorize
def log_expit(x):
    return x - mp.log1p(mp.exp(x)) if x <= 0 else -mp.log1p(1/mp.exp(x))


def _boxcox_scalar(x, lmbda):
    """
    y = (x**lmbda - 1) / lmbda  if lmbda != 0
        log(x)                  if lmbda == 0
    """
    if x < 0:
        return mp.nan
    if x == 0:
        return lmbda*mp.inf
    if lmbda != 0:
        return mp.powm1(x, lmbda) / lmbda
    else:
        return mp.log(x)


@vectorize
def boxcox(x, lmbda):
    return _boxcox_scalar(x, lmbda)


@vectorize
def boxcox1p(x, lmbda):
    if x == 0:
        # Handle x = 0 separately to avoid blow up in extra_dps calculation.
        return mp.zero
    extra_dps = max(0, int(mp.ceil(-mp.log10(abs(x)))))
    with mp.workdps(mp.dps + extra_dps):
        return _boxcox_scalar(mp.one + x, lmbda)


# I forgot this could be solved analytically
# @vectorize
# def inv_boxcox(y, lmbda):
#     if xp.isnan(lmbda):
#         return mp.nan

#     if lmbda > 0:
#         if y < -1/lmbda:
#             return mp.nan
#         if y == -1/lmbda:
#             return 0
#         if y == mp.inf:
#             return mp.inf

#     if lmbda < 0:
#         if y > -1/lmbda:
#             return mp.nan
#         if y == -1/lmbda:
#             return mp.inf
#         if y == -mp.inf:
#             return 0

#     def f(x):
#         return _boxcox_scalar(x, lmbda) - y

#     b = 1
#     while f(b) <= 0:
#         print(b, f(b))
#         b = b*2
#     # illinois returns NaN when f(0) = -inf
#     # bisect fails when f(b) == 0
#     return mp.findroot(f, (0, b), solver='bisect', maxsteps=1000)


# TODO: add all features of SciPy version; until then, use SciPy implementation
# def logsumexp(a, axis=None, b=None):
#     # As far as I know, logsumexp is to avoid overflow, not to improve precision.
#     # mpmath doesn't overflow, so naive implementation should be OK.
#     return xp.log(xp.sum(b*xp.exp(a), axis=axis))


def ive(v, z):
    return iv(v, z) * xp.exp(-xp.abs(xp.real(z)))


def i0e(x):
    return ive(0, x)


def i1e(x):
    return ive(1, x)


def kve(v, z):
    return kv(v, z) * xp.exp(z)


def k0e(x):
    return kve(0, x)


def k1e(x):
    return kve(1, x)


def chdtr(v, x):
    return gammainc(v / 2, x / 2)


def chdtrc(v, x):
    return gammaincc(v / 2, x / 2)


def chdtri(v, p):
    return 2*gammainccinv(v / 2, p)


def stdtr(df, t):
    df, t = promote(df, t, atleast=float)
    x = df / (t**2 + df)
    p = betainc(df/2, mp.one/2, x)/2
    return xp.where(t < 0, p, mp.one - p)


@vectorize
def stdtrit(df, p):
    if not ((df >= 0) and (p >= 0 and p <= 1)):
        return mp.nan

    if p == 0:
        return -mp.inf
    elif p == 1:
        return mp.inf

    def f(t):
        x = df / (t**2 + df)
        cdf = mp.betainc(df/2, mp.one/2, 0, x, regularized=True)/2
        res = (cdf if t < 0 else mp.one - cdf) - p
        return res

    b = 1
    while not (f(-b) < 0 < f(b)):
        b = b*2
    return mp.findroot(f, (-b, b), solver='illinois', maxsteps=1000)


def entr(x):
    return xp.where(x >= 0, -xlogy(x, x), -mp.inf)


def rel_entr(x, y):
    return xp.where((x >= 0) & (y >= 0), xlogy(x, x) - xlogy(x, y), mp.inf)


def log_gammainc(a, x):  # TODO: add tests when public in SciPy 2.0
    exp_res = gammainc(a, x)
    return xp.where(exp_res < 0.5, xp.log(exp_res), xp.log1p(-gammaincc(a, x)))


def log_gammaincc(a, x):  # TODO: add tests when public in SciPy 2.0
    exp_res = gammaincc(a, x)
    return xp.where(exp_res < 0.5, xp.log(exp_res), xp.log1p(-gammainc(a, x)))


@vectorize
def gammaincinv(a, y):
    if not ((a >= 0) and (y >= 0 and y <= 1)):
        return mp.nan

    if y == 0:
        return 0

    if y == 1:
        return mp.inf

    if a == mp.inf or a == 0:
        return mp.nan

    def f(x):
        return mp.gammainc(a, a=0, b=x, regularized=True) - y
    b = 1
    while f(b) < 0:
        b = b*2
    return mp.findroot(f, (0, b), solver='illinois', maxsteps=1000)


@vectorize
def gammainccinv(a, y):
    if not ((a >= 0) and (y >= 0 and y <= 1)):
        return mp.nan

    if y == 0:
        return mp.inf

    if y == 1:
        return 0

    if a == mp.inf or a == 0:
        return mp.nan

    def f(x):
        return mp.gammainc(a, a=x, b=mp.inf, regularized=True) - y
    left = 1
    while f(left) > 0:
        left = left*2
    return mp.findroot(f, (0, left), solver='illinois', maxsteps=1000)



def fresnel(x):
    return fresnels(x), fresnelc(x)


# others to be added
# chndtr
# chndtrix
# tklmbda
# inv_boxcox
# inv_boxcox1p
# kolmogorov, smirnov


# generate rough documentation
_preface = ["The following is the documentation for the corresponding "
            "attribute of `scipy.special`.",
            "MPArray behavior is the same except that the calculation is "
            "carried out in the appropriate precision.\n\n"]
_preface = "\n".join(_preface)
function_names = list(sys.modules[__name__].__dict__.keys())
for key in function_names:
    if key in _imports or '_' in key:
        continue
    special_doc = getattr(special, key).__doc__
    sys.modules[__name__].__dict__[key].__doc__ = _preface + special_doc
