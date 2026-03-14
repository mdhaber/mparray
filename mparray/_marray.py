import collections
import inspect
import sys
import types
import numpy as np
import numpy.testing
import mpmath
from mpmath import mp
import functools


class MPArray:

    def __init__(self, obj, *, dtype=None, device=None, copy=None):
        # TODO: fast path if input is MPArray and nothing is changing

        if isinstance(obj, MPArray):
            data = np.asarray(obj._data, device=device, copy=copy)
        elif isinstance(obj, bool):
            data = np.asarray(obj)
        elif isinstance(obj, (int, mp.mpf, mp.mpc)):
            data = np.asarray(obj, dtype=object, device=device, copy=copy)
        else:
            data = np.asarray(obj, device=device, copy=copy)
            dtype = data.dtype if dtype is None else dtype

        dtype = _get_dtype(obj) if dtype is None else _get_dtype(dtype)
        shape = data.shape

        dtype_ = object
        if np.isdtype(dtype, 'bool'):
            type_ = bool
            dtype_ = np.bool
        if np.isdtype(dtype, 'integral'):
            type_ = int
        elif np.isdtype(dtype, 'real floating'):  # TODO: fix for bool input
            type_ = lambda x: mp.mpf(x) if isinstance(x, (mp.mpf, mp.mpc)) else mp.mpf(str(x))
        elif np.isdtype(dtype, 'complex floating'):  # TODO: fix for complex inf/nan
            type_ = lambda x: mp.mpc(x) if isinstance(x, (mp.mpf, mp.mpc)) else mp.mpc(str(x))

        data = np.asarray([type_(el) for el in data.ravel()], dtype=dtype_)
        data = np.reshape(data, shape)

        self._data = data
        self._dtype = dtype
        self._device = data.device
        self._ndim = data.ndim
        self._shape = data.shape
        self._size = data.size

    __array_priority__ = 1  # make reflected operators work with NumPy

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def ndim(self):
        return self._ndim

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

    def __array_namespace__(self, api_version=None):
        if api_version is None or api_version == '2024.12':
            return mod
        else:
            message = (f"MPArray interface for Array API version '{api_version}' "
                       "is not implemented.")
            raise NotImplementedError(message)

    def _call_super_method(self, method_name, *args, **kwargs):
        method = getattr(self._data, method_name)
        args = [_get_data(arg) for arg in args]
        return method(*args, **kwargs)

    def _validate_key(self, key):
        if isinstance(key, tuple):
            return tuple(self._validate_key(key_i) for key_i in key)

        if isinstance(key, MPArray):
            if np.isdtype(key.dtype, 'integral'):
                return key._data.tolist()
            return key._data

        return key

    # ## Indexing ##

    def __getitem__(self, key):
        key = self._validate_key(key)
        return asarray(self._data[key], dtype=self.dtype, device=self.device)

    def __setitem__(self, key, other):
        key = self._validate_key(key)
        other = asarray(other, dtype=self.dtype, device=self.device)
        self._data.__setitem__(key, other._data)

    def __iter__(self):
        return iter(self._data)

    def __deepcopy__(self, memo=None):
        return asarray(self, copy=True)

    # ## Visualization ##
    def __repr__(self):
        s = repr(self._data)
        s = s.replace('array', 'MPArray')
        return s.replace("dtype=object", f"dtype={self.dtype}")

    def __str__(self):
        # TODO: refine to show full precision?
        return str(np.asarray(self._data, dtype=self.dtype))

    # ## Linear Algebra Methods ##
    def __matmul__(self, other):
        return mod.matmul(self, other)

    def __imatmul__(self, other):
        res = mod.matmul(self, other)
        self._data[...] = res.data[...]
        return

    def __rmatmul__(self, other):
        other = asarray(other)
        return mod.matmul(other, self)

    ## Attributes ##

    @property
    def T(self):
        return asarray(self._data.T, dtype=self.dtype)

    @property
    def mT(self):
        return mod.matrix_transpose(self)

    # dlpack
    def __dlpack_device__(self):
        return self._data.__dlpack_device__()

    def __dlpack__(self):
        # really not sure how to define this
        return self._data.__dlpack__()

    def to_device(self, device, /, *, stream=None):
        self._data = self._data.to_device(device, stream=stream)

    def __index__(self):
        if self.shape == () and np.isdtype(self.dtype, 'integral'):
            return self._data[()]
        else:
            message = "Only integer scalar arrays can be converted to a scalar index."
            raise ValueError(message)


## Methods ##

# Methods that return the result of a unary operation as an array
unary_names = (['__abs__', '__invert__', '__neg__', '__pos__'])
for name in unary_names:
    def fun(self, name=name):
        data = self._call_super_method(name)
        return asarray(data, dtype=self.dtype)
    setattr(MPArray, name, fun)

# Methods that return the result of a unary operation as a Python scalar
unary_names_py = ['__bool__', '__complex__', '__float__', '__int__']
for name in unary_names_py:
    def fun(self, name=name):
        return self._call_super_method(name)
    setattr(MPArray, name, fun)

# Methods that return the result of an elementwise binary operation
binary_names = ['__add__', '__sub__', '__and__', '__eq__', '__ge__', '__gt__',
                '__le__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__',
                '__or__', '__pow__', '__rshift__', '__sub__', '__truediv__',
                '__xor__'] + ['__divmod__', '__floordiv__']
# Methods that return the result of an elementwise binary operation (reflected)
rbinary_names = ['__radd__', '__rand__', '__rdivmod__', '__rfloordiv__',
                 '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__rpow__',
                 '__rrshift__', '__rsub__', '__rtruediv__', '__rxor__']
ensure_output_dtype = ['__and__', '__eq__', '__ge__', '__gt__', '__le__', '__lt__',
                       '__ne__', '__or__', '__xor__', '__rand__', '__ror__', '__rxor__']
for name in binary_names + rbinary_names:
    def fun(self, other, name=name):
        self, other = _promote(self, other)
        dtype = None if name in ensure_output_dtype else self.dtype
        data = self._call_super_method(name, other)
        return asarray(data, dtype=dtype)
    setattr(MPArray, name, fun)

# In-place methods
desired_names = ['__iadd__', '__iand__', '__ifloordiv__', '__ilshift__',
                 '__imod__', '__imul__', '__ior__', '__ipow__', '__irshift__',
                 '__isub__', '__itruediv__', '__ixor__']
for name in desired_names:
    def fun(self, other, name=name, **kwargs):
        other = astype(other, self.dtype)
        self._call_super_method(name, other)
        return self
    setattr(MPArray, name, fun)

mod = sys.modules[__name__].__dict__

## Constants ##
constant_names = ['e', 'inf', 'nan', 'pi']
for name in constant_names:
    mod[name] = mp.mpf(getattr(mpmath, name))
newaxis = np.newaxis


## Creation Functions ##
def asarray(obj, /, *, dtype=None, device=None, copy=None):
    return MPArray(obj, dtype=dtype, device=device, copy=copy)


creation_functions = ['arange', 'empty', 'eye', 'from_dlpack',
                      'linspace', 'ones', 'zeros']
creation_functions_like = ['empty_like', 'ones_like', 'zeros_like']
# `full` and `full_like` created separately
#  'tril', 'triu', 'meshgrid' handled with array manipulation functions
for name in creation_functions:
    def fun(*args, name=name, **kwargs):
        data = getattr(np, name)(*args, **kwargs)
        return asarray(data)
    mod[name] = fun

for name in creation_functions_like:
    def fun(x, /, name=name, **kwargs):
        name = name.split("_")[0]
        kwds = dict(shape=x.shape, dtype=x.dtype, device=x.device)
        kwds.update(kwargs)
        shape = kwds.pop('shape')
        data = getattr(np, name)(shape, **kwds)
        return asarray(data)
    mod[name] = fun


def full(shape, fill_value, *, dtype=None, device=None):
    dtype = result_type(fill_value) if dtype is None else dtype
    res = mod.ones(shape, dtype=dtype, device=device)
    res[...] = fill_value
    return res


def full_like(x, /, fill_value, **kwargs):
    kwds = dict(shape=x.shape, dtype=x.dtype, device=x.device)
    kwds.update(kwargs)
    shape = kwds.pop('shape')
    return mod.full(shape, fill_value, **kwargs)


## Data Type Functions and Data Types ##
def result_type(*args):
    return np.result_type(*(_get_dtype(arg) for arg in args if arg is not None))


dtype_fun_names = ['can_cast', 'finfo', 'iinfo']
for name in dtype_fun_names:
    # TODO: consider these more carefully
    def fun(*args, name=name, **kwargs):
        args = [_get_dtype(arg) for arg in args]
        return getattr(np, name)(*args, **kwargs)
    mod[name] = fun

dtype_names = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
               'uint32', 'uint64', 'float32', 'float64', 'complex64', 'complex128',
               'isdtype']  # not really a dtype, but OK to treat it like one here
inspection_fun_names = ['__array_namespace_info__']
version_attribute_names = ['__array_api_version__']
for name in (dtype_names + inspection_fun_names + version_attribute_names):
    mod[name] = getattr(np, name)

def astype(x, dtype, /, *, copy=True, device=None):
    if device is None and not copy and dtype == x.dtype:
        return x
    # TODO: take care of copy=False error if impossible to satisfy
    return asarray(x, dtype=dtype, device=device, copy=copy)


## Elementwise Functions ##
# TODO: fix `logical_` functions for non-boolean dtype
elementwise_numpy = ['equal', 'greater', 'greater_equal', 'less', 'less_equal',
                     'logical_and', 'logical_not', 'logical_or', 'logical_xor',
                     'not_equal']
for name in elementwise_numpy:
    def fun(*args, name=name, **kwargs):
        args = (_get_data(arg) for arg in args)
        return asarray(getattr(np, name)(*args, **kwargs))
    mod[name] = fun

# TODO: fix `bitwise_` functions for inappropriate dtypes
elementwise_no_dtype = ['abs', 'bitwise_and', 'bitwise_left_shift', 'bitwise_invert',
                        'bitwise_or', 'bitwise_right_shift', 'bitwise_xor', 'negative',
                        'positive', 'square']
elementwise_promote_numpy = ['add', 'remainder', 'pow', 'multiply',
                             'maximum', 'minimum', 'subtract']
for name in elementwise_no_dtype + elementwise_promote_numpy:
    def fun(*args, name=name, **kwargs):
        args = _promote(*args)
        dtype = args[0].dtype
        args = (_get_data(arg) for arg in args)
        return asarray(getattr(np, name)(*args, **kwargs), dtype=dtype)
    mod[name] = fun

mp.reciprocal = lambda x: 1 / x  # lazy!
mp.logaddexp = lambda x, y: mp.log(mp.exp(x) + mp.exp(y))
mp.imag = lambda x: x.imag
mp.real = lambda x: x.real
mp.trunc = lambda x: mp.floor(x) if x > 0 else mp.ceil(x)
mp.round = lambda x: mp.nint(x)
elementwise_mp = ['acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'cos',
                  'cosh', 'divide', 'exp', 'expm1', 'hypot', 'log', 'log1p', 'log2',
                  'log10', 'logaddexp', 'reciprocal', 'sin', 'sinh', 'sqrt', 'tan',
                  'tanh']
elementwise_mp_float = ['ceil', 'conj', 'floor', 'imag', 'real', 'round', 'trunc']
for name in elementwise_mp + elementwise_mp_float:
    def fun(*args, name=name, **kwargs):
        atleast = bool if name in elementwise_mp_float else float
        args = _promote(*args, atleast=atleast)

        if name in elementwise_mp_float and np.isdtype(args[0].dtype, ('bool', 'integral')):
            return args[0]  # TODO: fix imag

        data = (_get_data(arg) for arg in args)
        out = np.vectorize(getattr(mp, name), otypes=[object])(*data, **kwargs)
        # TODO: preserve complex output dtype for funcs like acos
        return asarray(out, dtype=args[0].dtype)
    mod[name] = fun


elementwise_is = ['isfinite', 'isinf', 'isnan']
for name in elementwise_is:
    def fun(arg, name=name, **kwargs):
        arg = asarray(arg)
        if np.isdtype(arg.dtype, ('bool', 'integral')):
            return mod.full_like(arg, name=='isfinite', dtype=np.bool)

        out = np.vectorize(getattr(mp, name), otypes=[bool])(_get_data(arg), **kwargs)
        return asarray(out, dtype=np.bool)
    mod[name] = fun


def floor_divide(x1, x2, /):
    x1, x2 = _promote(x1, x2)
    return asarray(mod.floor(x1 / x2), dtype=x1.dtype)


def sign(x, /):
    x = asarray(x)
    if mod.isdtype(x.dtype, ('bool', 'integral')):
        return asarray(np.sign(x._data), dtype=x.dtype)
    return asarray(_vectorize(mp.sign, otypes=[object])(x), dtype=x.dtype)


def signbit(x, /):
    return x < 0  # Python int/mp.mpf don't have -0 or signed NaNs


def copysign(x1, x2, /):
    return abs(x1) * sign(x2)


def nextafter(x1, x2, /):
    x1, x2 = asarray(x1), asarray(x2)
    inc = 10 ** (mod.floor(mod.log10(x1)) - mp.dps)  # TODO: defined for other types
    return x1 + mod.sign(x2 - x1)*inc


def clip(x, /, min=None, max=None):
    x, min, max = _promote(x, min, max)
    dtype = x.dtype
    x, min, max = _get_data(x, min, max)
    out = np.clip(x, min=min, max=max)
    return asarray(out, dtype=dtype)

## Indexing Functions
take_like = ['take', 'take_along_axis']
for name in take_like:
    def fun(x, indices, name=name, **kwargs):
        x = asarray(x)
        dtype = x.dtype
        x, indices = _get_data(x, indices)
        indices = np.astype(indices, np.int64)
        return asarray(getattr(np, name)(x, indices, **kwargs), dtype=dtype)
    mod[name] = fun

## Inspection ##
# Included with dtype functions above

## Linear Algebra Functions ##
linalg_names = ['matmul', 'tensordot', 'vecdot']
for name in linalg_names:
    def fun(x1, x2, /, name=name, **kwargs):
        x1, x2 = _promote(x1, x2)
        dtype = x1.dtype
        x1, x2 = _get_data(x1, x2)
        out = getattr(np, name)(x1, x2)
        return asarray(out, dtype=dtype)
    mod[name] = fun

matrix_transpose = lambda x: asarray(_get_data(x).mT, dtype=x.dtype)

## Manipulation Functions ##
output_arrays = {'broadcast_arrays', 'unstack', 'meshgrid'}

manip_array_in_out = ['broadcast_arrays', 'meshgrid']
for name in manip_array_in_out:
    def fun(*args, name=name, **kwargs):
        args = tuple(_promote(*args))
        dtype = args[0].dtype
        res = getattr(np, name)(*_get_data(*args), **kwargs)
        return tuple(asarray(resi, dtype=dtype) for resi in res)
    mod[name] = fun

manip_tuple_in = ['concat', 'stack']
for name in manip_tuple_in:
    def fun(args, name=name, **kwargs):
        args = tuple(_promote(*args))
        dtype = args[0].dtype
        res = getattr(np, name)(tuple(_get_data(*args)), **kwargs)
        return asarray(res, dtype=dtype)
    mod[name] = fun

manip_names = ['broadcast_to', 'expand_dims', 'flip', 'moveaxis', 'permute_dims',
               'reshape', 'roll', 'squeeze', 'tile', 'tril', 'triu']
for name in manip_names:
    def fun(x, *args, name=name, **kwargs):
        x = asarray(x)
        res = getattr(np, name)(_get_data(x), *args, **kwargs)
        return asarray(res, dtype=x.dtype)
    mod[name] = fun


def repeat(x, repeats, /, *, axis):
    x = asarray(x)
    repeats = np.asarray(_get_data(repeats), dtype=np.int64)
    res = np.repeat(x._data, repeats, axis=axis)
    return asarray(res, dtype=x.dtype)


def unstack(x, /, *, axis):
    x = asarray(x)
    res = np.unstack(x._data, axis=axis)
    return tuple(asarray(resi, dtype=x.dtype) for resi in res)


broadcast_shapes = np.broadcast_shapes

## Searching Functions
def searchsorted(x1, x2, /, *, side='left', sorter=None):
    x1, x2 = _promote(x1, x2)
    x1, x2 = _get_data(x1, x2)
    j = np.searchsorted(x1, x2, side=side, sorter=sorter)
    return asarray(j)


def nonzero(x, /):
    x = asarray(x)
    res = np.nonzero(x._data)
    return tuple(asarray(resi) for resi in res)


def where(condition, x1, x2, /):
    condition = asarray(condition)
    x1, x2 = _promote(x1, x2)
    data = np.where(condition._data, x1._data, x2._data)
    return asarray(data, dtype=x1.dtype)

# Defined below, in Statistical Functions
# argmax
# argmin
# count_nonzero

## Set Functions ##
unique_names = ['unique_values', 'unique_counts', 'unique_inverse', 'unique_all']
for name in unique_names:
    def fun(x, /, name=name):
        x = asarray(x)
        res = getattr(np, name)(x._data)
        if name == 'unique_values':
            return asarray(res, dtype=x.dtype)

        fields = res._fields
        name_tuple = res.__class__.__name__
        result_class = collections.namedtuple(name_tuple, fields)

        result_list = []
        for res_i, field_i in zip(res, fields):
            dtype = x._dtype if field_i == 'values' else None
            result_list.append(asarray(res_i, dtype=dtype))
        return result_class(*result_list)
    mod[name] = fun

## Sorting Functions ##
sort_names = ['sort', 'argsort']
for name in sort_names:
    def fun(x, /, *, name=name, axis=-1, descending=False, stable=True):
        x = asarray(x)
        res = getattr(np, name)(x._data)
        return asarray(res, dtype=x.dtype if name == 'sort' else None)
    mod[name] = fun

## Statistical Functions and Utility Functions ##
statistical_names_float = ['mean', 'var', 'std']
statistical_names_dtype = ['max', 'min', 'sum', 'prod',
                           'cumulative_sum', 'cumulative_prod', 'diff']
statistical_names_none = ['argmax', 'argmin', 'count_nonzero', 'all', 'any']
for name in statistical_names_float + statistical_names_dtype + statistical_names_none:
    def fun(x, *args, name=name, **kwargs):
        dtype = kwargs.pop('dtype', float if name in statistical_names_float else bool)
        x, = _promote(x, atleast=dtype)  # TODO: follow standard precisely?
        res = getattr(np, name)(x._data, *args, **kwargs)
        return asarray(res, dtype=None if name in statistical_names_none else x.dtype)
    mod[name] = fun


preface = ["The following is the documentation for the corresponding "
           f"attribute of NumPy.",
           "MPArray behavior is the same except that the calculation is "
           "carried out in the appropriate precision.\n\n"]
preface = "\n".join(preface)
mod_keys = list(mod.keys())
for attribute in mod_keys:
    # Add documentation if it is not already present
    if mod[attribute].__doc__:
        continue

    np_attr = getattr(np, attribute, None)
    mod_attr = mod.get(attribute, None)
    if np_attr is not None and mod_attr is not None:

        if hasattr(np_attr, "__doc__"):
            try:
                np_doc = getattr(np_attr, "__doc__")
                mod[attribute].__doc__ = preface + np_doc
            except (AttributeError, TypeError):
                pass

        try:
            mod_attr.__signature__ = inspect.signature(np_attr)
        except (ValueError, TypeError):
            pass

        try:
            mod_attr.__name__ = np_attr.__name__
        except (AttributeError, TypeError):
            pass


def _xinfo(x):
    np = x._np
    if np.isdtype(x.dtype, 'integral'):
        return np.iinfo(x.dtype)
    elif np.isdtype(x.dtype, 'bool'):
        binfo = dataclasses.make_dataclass("binfo", ['min', 'max'])
        return binfo(min=False, max=True)
    else:
        return np.finfo(x.dtype)


def _get_data(*args):
    if len(args) == 1:
        x = args[0]
        return x._data if isinstance(x, MPArray) else x
    return tuple(_get_data(arg) for arg in args)


def _get_dtype(x):
    if isinstance(x, bool):  # x in [True, False]:  # TODO: fix overriding built-in bool
        return np.bool
    elif isinstance(x, int):
        return np.int64
    elif isinstance(x, mp.mpf):
        return np.float64
    elif isinstance(x, mp.mpc):
        return np.complex128
    elif x is np.bool:
        return x
    elif isinstance(x, type) and x not in {bool, int, float, complex}:
        return x(0).dtype

    return getattr(x, "dtype", x)


def _promote(*args, atleast=bool):
    dtype = result_type(*args, atleast)
    return tuple((astype(arg, dtype) if arg is not None else arg) for arg in args)


def _vectorize(f):

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        args = list(_promote(*args, atleast=float))
        data = (_get_data(arg) for arg in args)
        out = np.vectorize(f, otypes=[object])(*data, **kwargs)
        # TODO: preserve complex output dtype for funcs like acos
        return asarray(out, dtype=args[0].dtype)

    return wrapped
