import sys
import array_api_compat.numpy as np
import numpy.testing
from mpmath import mp
import functools

# TODO:
#  add test suite
#  allow each element of array to be different type?
#  error when special function is not available


def mptype(x):
    el = np.asarray(x).ravel()[0]

    if isinstance(el, int):
        return int
    if isinstance(el, mp.mpf):
        return mp.mpf
    if isinstance(el, mp.mpc):
        return mp.mpc
    elif np.issubdtype(el.dtype, np.integer):
        return int
    if np.issubdtype(el.dtype, np.floating):
        return mp.mpf
    elif np.issubdtype(el.dtype, np.complexfloating):
        return mp.mpc
    else:
        raise ValueError("Unrecognized type")


def nptype(x):
    el = np.asarray(x).ravel()[0]

    if isinstance(el, int):
        return np.int64
    if isinstance(el, mp.mpf):
        return np.float64
    if isinstance(el, mp.mpc):
        return np.complex128
    elif np.issubdtype(el.dtype, np.integer):
        return np.int64
    if np.issubdtype(el.dtype, np.floating):
        return np.float64
    elif np.issubdtype(el.dtype, np.complexfloating):
        return np.complex128
    else:
        raise ValueError("Unrecognized type")


# Array Object (Operators, Attributes, and Methods)
class mparray(np.ndarray):
    def __new__(cls, data):
        if isinstance(data, cls):
            return data

        data = np.asarray(data, dtype=object)

        message = ("This library is for getting real work done. "
                   "Pathological, useless input is not supported.")
        if data.size == 0:
            raise ValueError(message)

        shape = data.shape
        data = data.ravel()
        el = data[0]
        dtype = type(el)

        if np.issubdtype(dtype, np.floating):
            data = data.astype(np.float64)
            dtype = mp.mpf
        elif np.issubdtype(dtype, np.complexfloating):
            data = data.astype(np.complex128)
            dtype = mp.mpc
        elif not isinstance(el, int) and np.issubdtype(dtype, np.integer):
            data = data.astype(np.int64)
            dtype = int

        data = np.asarray([dtype(el) for el in data], dtype=object)
        return data.reshape(shape).view(cls)

    def __init__(self, data):
        self._dtype = type(self.ravel()[0])

    def __floordiv__(self, other):
        return np.floor(self/other)

    def __rfloordiv__(self, other):
        return np.floor(other/self)

    def __repr__(self):
        r = super().__repr__()
        if self.dtype == np.dtype(object):
            return r[:-15] + ')'
        else:
            return r

def asarray(*args, dtype=None, **kwargs):
    res = np.asarray(*args, dtype=object, **kwargs)

    shape = res.shape
    res = res.ravel()
    for i in range(res.size):
        t = mptype(res[i]) if dtype is None else mptype(dtype(res[i]))
        res[i] = t(res[i])
    res = res.reshape(shape)

    return mparray(res)
asarray.__doc__ = np.asarray.__doc__

def astype(x, dtype, *, copy=True):
    return asarray(x, dtype=dtype, copy=copy)
astype.__doc__ = np.astype.__doc__

def vectorize(f):
    vf = np.vectorize(f)

    @functools.wraps(f)
    def wrapped(x, *args, **kwargs):
        out = vf(x, *args, **kwargs)
        return asarray(out)

    return wrapped

def ensure_mp(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        out = f(*args, **kwargs)
        return asarray(out)

    return wrapped


sys.modules[__name__].__dict__['mpf'] = mp.mpf
sys.modules[__name__].__dict__['mpc'] = mp.mpc

def dps(n):
    """Set mpmath.dps to the specified number of digits"""
    mp.dps = n

# Constants
constants = """
e
inf
nan
pi
"""

for c in constants.split():
    sys.modules[__name__].__dict__[c] = asarray(mp.mpf(getattr(mp, c)))

newaxis = np.newaxis

# Creation Functions
creation_funcs = """
arange
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
    sys.modules[__name__].__dict__[f] = (
        lambda *args, f=f, dtype=None, **kwargs: asarray(getattr(np, f)(*args, dtype=dtype, **kwargs), dtype=dtype)
    )

# Data type functions defined with other_funcs
dtype_funcs = """
can_cast
finfo
iinfo
isdtype
result_type
"""

for f in dtype_funcs.split():
    # arguably results should be converted to mpfarray if they weren't already?
    sys.modules[__name__].__dict__[f] = getattr(np, f)

# need to define these manually... or not
# bitwise_and
# bitwise_left_shift
# bitwise_invert
# bitwise_or
# bitwise_right_shift
# bitwise_xor
# logical_xor


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
        sys.modules[__name__].__dict__[f] = vectorize(getattr(mp, f))
    except AttributeError:
        sys.modules[__name__].__dict__[f] = ensure_mp(getattr(np, f))

real = vectorize(mp.re)
imag = vectorize(mp.im)
round = vectorize(mp.nint)

@ensure_mp
def floor_divide(x, y):
    return x // y
floor_divide.__doc__ = np.floor_divide.__doc__

@vectorize
def log2(x):
    return mp.log(x) / mp.log(2)
log2.__doc__ = np.log2.__doc__

@vectorize
def logaddexp(a, b):
    # IIUC, logaddexp avoids overflow but doesn't improve precision.
    # mpmath doesn't overflow, so naive implementation should be OK.
    return mp.log(mp.exp(a) + mp.exp(b))
logaddexp.__doc__ = np.logaddexp.__doc__

@vectorize
def trunc(x):
    return mp.floor(x) if x >= 0 else mp.ceil(x)
trunc.__doc__ = np.trunc.__doc__

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

for f in (other_funcs).split():
    # arguably results should be converted to mpfarray if they weren't already?
    sys.modules[__name__].__dict__[f] = getattr(np, f)

def asnumpy(x):
    return np.asarray(x, dtype=nptype(x))


def assert_allclose(res, ref,
                    check_type=True, check_shape=True, check_trivial=False):

    if check_type:
        assert_mptype(res)
        message = "`res` mptype does not match `ref` mptype."
        resb, refb = np.broadcast_arrays(res, ref)
        for resi, refi in zip(resb.ravel(), refb.ravel()):
            assert mptype(resi) == mptype(refi), message

    if check_trivial:
        assert_nontrival(res)

    if check_shape:
        message = "`res.shape` != `ref.shape`."
        assert res.shape == ref.shape, message

    numpy.testing.assert_allclose(asnumpy(res), ref)


def assert_nontrival(res):
    res = asnumpy(res)
    message = "`res` contains trivial values."
    assert np.all(np.isfinite(res) & (res != 0) & (res != 1)), message


def assert_mptype(res):

    message = "`res` is not an mparray."
    assert isinstance(res, mparray), message

    for resi in res.ravel():
        message = "`res` dtype is not an mptype."
        assert (isinstance(resi, int)
                or isinstance(resi, mp.mpf)
                or isinstance(resi, mp.mpc)
                or isinstance(resi, mp.constant)), message
