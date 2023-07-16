import sys
import numpy as np
from mpmath import mp
# TODO:
#  add dtypes
#  add test suite
#  add documentation
#  improve pretty printing
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
