import sys
import array_api_compat.numpy as np
from mpmath import mp
import functools

# TODO:
#  add test suite
#  error when special function is not available

# Array Object (Operators, Attributes, and Methods)
class mparray(np.ndarray):
    def __new__(cls, data):
        data = np.asarray(data)
        dtype = data.dtype

        if not (np.issubdtype(dtype, np.number)
                or np.issubdtype(dtype, np.bool_)):
            return data.view(cls)


        if np.issubdtype(dtype, np.bool_):
            type_ = bool
        if np.issubdtype(dtype, np.floating):
            data = data.astype(np.float64)
            type_ = mp.mpf
        elif np.issubdtype(dtype, np.complexfloating):
            data = data.astype(np.complex128)
            type_ = mp.mpc
        elif np.issubdtype(dtype, np.integer):
            type_ = int

        shape = data.shape
        data = data.ravel()
        data = np.asarray([type_(x) for x in data], dtype=object)
        return data.reshape(shape).astype(object, copy=False).view(cls)

    def __array_finalize__(self, obj):
        self.dtype = np.asarray(object()).dtype

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

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, val):
        self._dtype = val

def asarray(*args, dtype=None, **kwargs):
    nparr = np.asarray(*args, **kwargs)

    if dtype is None:
        ok_dtype = (np.issubdtype(nparr.dtype, np.number) or
                    np.issubdtype(nparr.dtype, np.bool_))
        dtype = nparr.dtype if ok_dtype else np.asarray(1.0).dtype

    arr = mparray(nparr)
    arr.dtype = dtype
    return arr
asarray.__doc__ = np.asarray.__doc__

def astype(x, dtype, *, copy=True):
    return asarray(x, dtype=dtype, copy=copy)
astype.__doc__ = np.astype.__doc__

def vectorize(f):
    vf = np.vectorize(f)

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        out = vf(*args, **kwargs)
        return asarray(out)

    return wrapped

def ensure_mp(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        out = f(*args, **kwargs)
        return asarray(out)

    return wrapped

# Constants
constants = """
e
inf
nan
pi
"""

for c in constants.split():
    sys.modules[__name__].__dict__[c] = asarray(getattr(mp, c))

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
        lambda *args, f=f, **kwargs: asarray(getattr(np, f)(*args, **kwargs))
    )

# Data Types to be defined
dtypes = """
bool
int8
int16
int32
int64
uint8
uint16
uint32
uint64
float32
float64
complex64
complex128
"""

for dtype in dtypes.split():
    sys.modules[__name__].__dict__[dtype] = getattr(np, dtype)

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
