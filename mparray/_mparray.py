import sys
import array_api_compat.numpy as np
from mpmath import mp
import functools

# TODO:
#  add test suite
#  error when special function is not available

class mybool(int):
    ndim = 0
    size = 1
    shape = tuple()
    dtype = None

class myint(int):
    ndim = 0
    size = 1
    shape = tuple()
    dtype = None

class mympf(mp.mpf):
    ndim = 0
    size = 1
    shape = tuple()
    dtype = None

class mympc(mp.mpc):
    ndim = 0
    size = 1
    shape = tuple()
    dtype = None

def _ensure_dtype(arr, dtype=None):

    data = arr.ravel()
    data.dtype = arr.dtype
    el = np.asarray(data[0])
    dtype = dtype or el.dtype

    if np.issubdtype(dtype, np.bool_):
        type_ = mybool
    elif np.issubdtype(dtype, np.floating):
        type_ = lambda x: mympf(np.float64(x))
    elif np.issubdtype(dtype, np.complexfloating):
        type_ = lambda x: mympc(np.complex128(x))
    elif np.issubdtype(dtype, np.integer):
        type_ = myint
    elif isinstance(el[()], mp.mpf):
        type_ = mympf
    elif isinstance(el[()], mp.mpc):
        type_ = mympc
    else:
        type_ = type(data[0])

    for i in range(data.size):
        data[i] = type_(data[i])
        data[i].dtype = dtype

    return dtype


# Array Object (Operators, Attributes, and Methods)
class mparray(np.ndarray):
    _dtype = None
    _finalized = False

    def __new__(cls, data):

        if isinstance(data, cls):
            return data

        return np.array(data, dtype=np.object_).view(cls)

    def __array_finalize__(self, obj):

        if isinstance(obj, type(self)):
            return

        if obj.size == 0:
            self.dtype = np.array([]).dtype
            return

        dtype = _ensure_dtype(obj)
        self.dtype = dtype
        self._finalized = True

    def __floordiv__(self, other):
        return np.floor(self/other)

    def __rfloordiv__(self, other):
        return np.floor(other/self)

    def __index__(self):
        if (np.issubdtype(self.dtype, np.integer)
                and self.ndim == 0 and self.size == 1):
            return int(self.item())
        else:
            super().__index__(self)

    # # how to make getitem return array sometimes but not others?
    # def __getitem__(self, item):
    #     if self.shape == item == tuple():
    #         return self
    #     else:
    #         el = super().__getitem__(item)
    #         if isinstance(el, mparray):
    #             el.dtype = self.dtype
    #         return el

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

def asarray(data, *args, dtype=None, **kwargs):
    data = mparray(data)
    dtype = _ensure_dtype(data, dtype or data.dtype)
    data._finalized = True
    data.dtype = dtype
    return data

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

# Constants
constants = """
e
inf
nan
pi
"""

for c in constants.split():
    sys.modules[__name__].__dict__[c] = mympf(getattr(mp, c))
    sys.modules[__name__].__dict__[c].dtype = np.float64

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
