import sys
import numpy as np
from mpmath import mp
# TODO:
#  add dtypes
#  add test suite
#  add documentation
#  improve pretty printing
#  add scipy special functions
#  put special functions in `special` namespace
#  error when special function is not available


# Array Object (Operators, Attributes, and Methods)
class mparray(np.ndarray):
    def __new__(cls, data):
        data = np.asarray(data)
        if np.issubdtype(data.dtype, object):
            return data

        # Only inexact dtypes can be converted to mpf/mpf
        data = data if np.issubdtype(data.dtype, np.inexact) else data.astype(np.float64)

        type_ = mp.mpf if np.issubdtype(data.dtype, np.floating) else mp.mpc
        shape = data.shape
        data = data.ravel()
        return np.asarray([type_(x) for x in data.ravel()]).reshape(shape).astype(object, copy=False)


# Constants
constants = """
e
inf
nan
pi
"""

for c in constants.split():
    sys.modules[__name__].__dict__[c] = getattr(mp, c)

newaxis = np.newaxis

# Creation Functions
creation_funcs = """
arange
asarray
empty
empty_like
eye
from_dlpack
full
full_like
linspace
meshgrid
ones
ones_like
tril
triu
zeros
zeros_like
"""

for f in creation_funcs.split():
    sys.modules[__name__].__dict__[f] = lambda *args, f=f, **kwargs: mparray(getattr(np, f)(*args, **kwargs))

# Data Type Functions need to be defined manually
# Data Types to be defined


# need to define these
# add
# bitwise_and
# bitwise_left_shift
# bitwise_invert
# bitwise_or
# bitwise_right_shift
# bitwise_xor
# divide
# equal
# floor_divide
# greater
# greater_equal
# less
# less_equal
# log2
# logaddexp
# logical_and
# logical_not
# logical_or
# logical_xor
# multiply
# negative
# not_equal
# positive
# pow
# remainder
# round
# square
# subtract
# trunc

abs = np.vectorize(abs)
real = np.vectorize(mp.re)
imag = np.vectorize(mp.im)

# Elementwise Functions
elementwise_funcs = """
abs
acos
acosh
add
asin
asinh
atan
atan2
atanh
bitwise_and
bitwise_left_shift
bitwise_invert
bitwise_or
bitwise_right_shift
bitwise_xor
ceil
conj
cos
cosh
divide
equal
exp
expm1
floor
floor_divide
greater
greater_equal
imag
isfinite
isinf
isnan
less
less_equal
log
log1p
log2
log10
logaddexp
logical_and
logical_not
logical_or
logical_xor
multiply
negative
not_equal
positive
pow
real
remainder
round
sign
sin
sinh
square
sqrt
subtract
tan
tanh
trunc
"""

for f in elementwise_funcs.split():
    try:
        sys.modules[__name__].__dict__[f] = np.vectorize(getattr(mp, f))
    except AttributeError:
        pass

# (Some of these need to be defined otherwise)
# Indexing Functions, Linear Algebra Functions
# Manipulation Functions, Searching Functions, Set Functions,
# Sorting Functions, Statistical Functions, Utility Functions
other_funcs = """
take
matmul
matrix_transpose
tensordot
vecdot
broadcast_arrays
broadcast_to
concat
expand_dims
flip
permute_dims
reshape
roll
squeeze
stack
argmax
argmin
nonzero
where
unique_all
unique_counts
unique_inverse
unique_values
argsort
sort
max
mean
min
prod
std
sum
var
all
any
"""

for f in other_funcs.split():
    # arguably results should be converted to mpfarray if they weren't already?
    sys.modules[__name__].__dict__[f] = lambda *args, f=f, **kwargs: getattr(np, f)(*args, **kwargs)


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



# Move these to a `special` namespace
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
    return mp.log(mp.ncdf(x))  # needs extra precision for large x


@np.vectorize
def betaln(x, y):
    return mp.log(mp.beta(x, y))  # may need extra precision


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


def logsumexp(x):
    # as far as I know, logsumexp is to avoid overflow,
    # not to improve precision, and mpmath avoids overflow
    return mp.log(exp(x).sum())  # noqa


def ive(v, z):
    return iv(v, z) * exp(-abs(real(z)))  # noqa


def i0e(x):
    return ive(0, x)


def i1e(x):
    return ive(1, x)


def kve(v, z):
    return kv(v, z) * exp(z)  # noqa


def k0e(x):
    return kve(0, x)


def k1e(x):
    return kve(1, x)


# others to be added
# gammaincinv
# gammainccinv
# chdtr
# chdtrc
# chdtri
# chndtr
# chndtrix
# stdtr
# stdtrit
# ndtri_exp
# tklmbda
# inv_boxcox
# inv_boxcox1p
# kolmogorov, smirnov, f
# erfcinv
# erfinv


