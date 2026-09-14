import inspect
import sys as sys
from collections import namedtuple

import numpy as np
from mpmath import mp
from scipy._lib._util import _apply_over_batch

import mparray as xp
from mparray._mparray import _get_data, _get_dtype, _promote

# add imported names to `_imports` to avoid altering their documentation and exposing
# as public members of `mparray.special`.
_imports = {'inspect', 'sys', 'np', 'mp', 'xp', 'name', 'fun', 'promote', 'mod'}

fun_names_args = [
    ('cholesky', [('x', 2)]),
    ('cross', [('x1', 1), ('x2', 1)]),
    ('det', [('x', 2)]),
    ('diagonal', [('x', 2)]),
    ('eig', [('x', 2)]),
    ('eigh', [('x', 2)]),
    # ('eigvals', [('x', 2)]),  # defined via eig
    # ('eigvalsh', [('x', 2)]),  # defined via eigh
    # ('inv', [('x', 2)]),  # defined via matrix_power
    # matmul is alias
    ('matrix_norm', [('x', 2)]),
    ('matrix_power', [('x', 2), ('n', 0)]),
    # ('matrix_rank', [('x', 2)]),  # defined via svd
    # matrix_transpose is alias
    ('outer', [('x1', 1), ('x2', 1)]),
    # ('pinv', [('x', 2)]),  # defined via svd
    ('qr', [('x', 2)]),
    # ('slogdet', [('x', 2)]),  # defined via det
    ('solve', [('x1', 2), ('x2', 2)]),
    ('svd', [('x', 2)]),
    # ('svdvals', [('x', 2)]),  # currently defined via svd
    # tensordot is alias
    # ('trace', [('x', 2)]),  # defined via diag
    # vecdot is alias
    ('vector_norm', [('x', 1)]),  # will need `axis` added
]

def _cholesky(A):
    data = _get_data(A)
    out = np.asarray(mp.cholesky(mp.matrix(data)).tolist(), dtype=object)
    dtype = xp.result_type(A.dtype, _get_dtype(out.ravel()[0]))
    return xp.asarray(out, dtype=dtype)


def _cross(x1, x2):
    out = np.cross(_get_data(x1), _get_data(x2))
    dtype = xp.result_type(x1.dtype, x2.dtype,_get_dtype(out.ravel()[0]))
    return xp.asarray(out, dtype=dtype)


def _det(A):
    data = _get_data(A)
    out = np.asarray(mp.det(mp.matrix(data)), dtype=object)
    dtype = xp.result_type(A.dtype, _get_dtype(out.ravel()[0]))
    return xp.asarray(out, dtype=dtype)


def _diagonal(A, offset=0):
    data = _get_data(A)
    out = np.diag(data, k=offset)
    dtype = xp.result_type(A.dtype, _get_dtype(out.ravel()[0]))
    return xp.asarray(out, dtype=dtype)


def _eig(A):
    data = _get_data(A)
    out1, out2 = mp.eig(mp.matrix(data))
    out1 = np.asarray(out1, dtype=object)
    out2 = np.asarray(out2.tolist(), dtype=object)
    dtype = xp.result_type(A.dtype, complex)
    return xp.asarray(out1, dtype=dtype), xp.asarray(out2, dtype=dtype)


def _eigh(A):
    data = _get_data(A)
    out1, out2 = mp.eigh(mp.matrix(data))
    out1 = np.asarray(out1.tolist(), dtype=object)[:, 0]
    out2 = np.asarray(out2.tolist(), dtype=object)
    dtype1 = xp.result_type(A.dtype, _get_dtype(out1.ravel()[0]))
    dtype2 = xp.result_type(A.dtype, _get_dtype(out2.ravel()[0]))
    return xp.asarray(out1, dtype=dtype1), xp.asarray(out2, dtype=dtype2)


def eigvals(A):
    return mod['eig'](A)[0]


def eigvalsh(A):
    return mod['eigh'](A)[0]


def inv(A):
    return mod['matrix_power'](A, -1)


def _matrix_norm(A, /, *, keepdims=False, ord='fro'):
    data = _get_data(A)
    if ord in {xp.inf, 1, 'fro'}:
        out = np.asarray(mp.mnorm(mp.matrix(data), ord), dtype=object)
    elif ord in {-2, 2, 'nuc'}:
        S = np.asarray(mp.svd(mp.matrix(data), compute_uv=False), dtype=object)
        if ord == 'nuc':
            out = np.sum(S)
        elif ord == 2:
            out = np.max(S)
        else:
            out = np.min(S)
        out = np.asarray(out, dtype=object)
    elif ord in {-1, -xp.inf}:
        axis = 0 if ord == -1 else 1
        out = np.asarray(np.min(np.sum(abs(data), axis=axis)), dtype=object)
    else:
        raise ValueError("Invalid norm order.")
    dtype = xp.result_type(A.dtype, _get_dtype(out.ravel()[0]))
    res = xp.real(xp.asarray(out, dtype=dtype))
    return res[..., xp.newaxis, xp.newaxis] if keepdims else res


def _matrix_power(A, n):
    data = _get_data(A)
    out = np.asarray((mp.matrix(data)**int(n)).tolist(), dtype=object)
    dtype = xp.result_type(A.dtype, _get_dtype(out.ravel()[0]))
    return xp.asarray(out, dtype=dtype)


def matrix_rank(x, /, *, rtol=None):
    x, = _promote(x, atleast=float)
    m, n = x.shape[-2:]
    rtol = max(m, n)*xp.finfo(x).eps if rtol is None else rtol
    S = xp.linalg.svdvals(x)
    tol = (rtol * xp.max(S, axis=-1))[..., xp.newaxis]
    return xp.count_nonzero(abs(S) > tol, axis=-1)


def _outer(x1, x2):
    out = np.outer(_get_data(x1), _get_data(x2))
    dtype = xp.result_type(x1.dtype, x2.dtype)
    return xp.asarray(out, dtype=dtype)


def pinv(x, /, *, rtol=None):
    x, = _promote(x, atleast=float)
    m, n = x.shape[-2:]
    rtol = max(m, n)*xp.finfo(x).eps if rtol is None else rtol
    U, S, Vh = xp.linalg.svd(x, full_matrices=False)
    tol = (rtol * xp.max(S, axis=-1))[..., xp.newaxis]
    S_ = xp.where(abs(S) < tol, 0, xp.reciprocal(S))
    S_ = xp.expand_dims(S_, axis=-2)
    return ((U * S_) @ Vh).mT


def _qr(A, /, *, mode):
    data = _get_data(A)
    out0, out1 = mp.qr(mp.matrix(data), mode=mode)
    out0 = np.asarray(out0.tolist(), dtype=object)
    out1 = np.asarray(out1.tolist(), dtype=object)
    return (xp.asarray(out0, dtype=A.dtype),
            xp.asarray(out1, dtype=A.dtype))


def _solve(A, b, /):
    data0 = _get_data(A)
    data1 = _get_data(b)
    out0 = mp.lu_solve(mp.matrix(data0), mp.matrix(data1))
    out0 = np.asarray(out0.tolist(), dtype=object)
    return xp.asarray(out0, dtype=A.dtype)


def _svd(A, /, *, full_matrices=True):
    data = _get_data(A)
    out0, out1, out2 = mp.svd(mp.matrix(data), full_matrices=full_matrices)
    out0 = np.asarray(out0.tolist(), dtype=object)
    out1 = np.asarray(out1.tolist(), dtype=object)[:, 0]
    out2 = np.asarray(out2.tolist(), dtype=object)
    return (xp.asarray(out0, dtype=A.dtype),
            xp.real(xp.asarray(out1, dtype=A.dtype)),
            xp.asarray(out2, dtype=A.dtype))


# def _svdvals(A):
#     data = _get_data(A)
#     out = mp.svd(mp.matrix(data), compute_uv=False)
#     out = np.asarray(out.tolist(), dtype=object)[:, 0]
#     return xp.real(xp.asarray(out, dtype=A.dtype))


def trace(A, /, *, offset=0, dtype=None):
    return xp.sum(mod['diagonal'](A, offset=offset), axis=-1, dtype=dtype)



mod = sys.modules[__name__].__dict__


for name, arg_data in fun_names_args:
    def fun(*args, name=name, arg_data=arg_data, **kwargs):
        if name == 'matrix_power':
            args = _promote(args[0], atleast=float)[0], args[1]
        elif name in {'trace', 'offset', 'outer', 'cross', 'diagonal'}:
            pass
        else:
            args = _promote(*args, atleast=float)
        fun = mod[f"_{name}"]
        return _apply_over_batch(*arg_data)(fun)(*args, **kwargs)
    mod[name] = fun


def cholesky(x, /, *, upper=False, cholesky=mod['cholesky']):
    if not x.size:  # temporary; use zero-size support in SciPy 2.0
        return x
    res = cholesky(x)
    return res.mT if upper else res


def cross(x1, x2, /, *, axis=-1, cross=mod['cross']):
    x1, x2 = xp.moveaxis(x1, axis, -1), xp.moveaxis(x2, axis, -1)
    if not x1.size or not x2.size:  # temporary; use zero-size support in SciPy 2.0
        res = xp.broadcast_arrays(x1, x2)[0]
    else:
        res = cross(x1, x2)
    return xp.moveaxis(res, -1, axis)


def det(x, /, *, det=mod['det']):
    if not x.size:  # temporary; use zero-size support in SciPy 2.0
        return xp.zeros(x.shape[:-2], dtype=x.dtype)
    return det(x)


def diagonal(x, /, *, offset=0, diagonal=mod['diagonal']):
    m, n = x.shape[-2:]
    if not x.size or abs(offset) >= min(m, n):
        k = max(min(m, n) - abs(offset), 0)
        return xp.zeros(x.shape[:-2]+(k,), dtype=x.dtype)
    return diagonal(x, offset=offset)


def eig(x, /, *, eig=mod['eig']):
    if not x.size:
        dtype, = _promote(x, atleast=complex)
        eigenvalues = xp.empty(x.shape[:-1], dtype=dtype)
        eigenvectors = xp.empty(x.shape, dtype=dtype)
    else:
        eigenvalues, eigenvectors = eig(x)
    return namedtuple("eig_result", ['eigenvalues', 'eigenvectors'])(
        eigenvalues=eigenvalues, eigenvectors=eigenvectors)


def eigh(x, /, *, eigh=mod['eigh']):
    if not x.size:
        dtype, = _promote(x, atleast=float)
        eigenvalues = xp.empty(x.shape[:-1], dtype=dtype)
        eigenvectors = xp.empty(x.shape, dtype=dtype)
    else:
        eigenvalues, eigenvectors = eigh(x)
    return namedtuple("eigh_result", ['eigenvalues', 'eigenvectors'])(
        eigenvalues=eigenvalues, eigenvectors=eigenvectors)


def matrix_power(x, n, /, *, matrix_power=mod['matrix_power']):
    x, = _promote(x, atleast=float)
    if not x.size:  # temporary; use zero-size support in SciPy 2.0
        return x
    return matrix_power(x, n)


def matrix_norm(x, /, *, keepdims=False, ord='fro', matrix_norm=mod['matrix_norm']):
    if not x.size:  # temporary; use zero-size support in SciPy 2.0
        res = xp.real(xp.zeros(x.shape[:-2], dtype=x.dtype))
        return res[..., xp.newaxis, xp.newaxis] if keepdims else res
    return matrix_norm(x, keepdims=keepdims, ord=ord)


def qr(x, /, *, mode='reduced', qr=mod['qr']):
    x, = _promote(x, atleast=float)
    if not x.size:
        batch_shape = x.shape[:-2]
        m, n = x.shape[-2:]
        k = min(m, n)
        q_core, r_core = ((m, k), (k, n)) if mode == 'reduced' else ((m, m), (m, n))
        Q = xp.empty(batch_shape + q_core, dtype=x.dtype)
        R = xp.empty(batch_shape + r_core, dtype=x.dtype)
    else:
        modes = {'reduced': 'skinny', 'complete': 'full'}
        if mode not in modes:
            raise ValueError(f'Unrecognized mode `{mode}`.')

        m, n = x.shape[-2:]
        if m < n:
            x_ = xp.zeros(x.shape[:-2] + (n, n), dtype=x.dtype)
            x_[..., :m, :] = x
            x = x_

        Q, R = qr(x, mode=modes[mode])

        if m < n:
            Q, R = Q[..., :m, :m], R[..., :m, :]

    return namedtuple("qr_result", ['Q', 'R'])(Q=Q, R=R)


def slogdet(x, /):
    x, = _promote(x, atleast=float)
    det = mod['det'](x)
    sign = xp.where(det == 0, 0, det / abs(det))
    logabsdet = xp.real(xp.astype(xp.log(abs(det)), x.dtype))
    return namedtuple("slogdet_result", ['sign', 'logabsdet'])(
        sign=sign, logabsdet=logabsdet)


def solve(x1, x2, /, *, solve=mod['solve']):
    x1, x2 = _promote(x1, x2, atleast=float)
    x2_ndim = x2.ndim

    x2 = x2[..., xp.newaxis]
    if x2_ndim > 1:
        # probably is a better way to do this!
        x2 = xp.expand_dims(x2, tuple(range(x1.ndim - x2_ndim)))
        x2 = xp.moveaxis(x2, -2, 0)
        x1, x2 = xp.broadcast_arrays(x1, x2)
        x2 = x2[..., :1]

    if not x1.size or not x2.size:
        x1, x2 = xp.broadcast_arrays(x1, x2)
        res = xp.empty(x2.shape[:-1], dtype=x2.dtype)
    else:
        res = solve(x1, x2)[..., 0]

    if x2_ndim > 1:
        res = xp.moveaxis(res, 0, -1)

    return res


def svd(x, /, *, full_matrices=True, svd=mod['svd']):
    if not x.size:
        x, = _promote(x, atleast=float)
        batch_shape = x.shape[:-2]
        m, n = x.shape[-2:]
        k = min(m, n)
        U_core = (m, m)
        Vh_core = (n, n)
        if not full_matrices:
            U_core, Vh_core = ((m, m), (k, n)) if m < n else ((m, k), (n, n))
        U = xp.empty(batch_shape + U_core, dtype=x.dtype)
        Vh = xp.empty(batch_shape + Vh_core, dtype=x.dtype)
        S = xp.real(xp.empty(batch_shape + (k,), dtype=x.dtype))
    else:
        U, S, Vh = svd(x, full_matrices=full_matrices)
    return namedtuple("svd_result", ['U', 'S', 'Vh'])(U=U, S=S, Vh=Vh)


def svdvals(x, /):
    return svd(x)[1]


def matrix_transpose(x, /):
    return xp.matrix_transpose(x)


def matmul(x1, x2, /):
    return xp.matmul(x1, x2)


def tensordot(x1, x2, /, *, axes=2):
    return xp.tensordot(x1, x2, axes=axes)


def vecdot(x1, x2, /, *, axis=-1):
    return xp.vecdot(x1, x2, axis=axis)


def vector_norm(x, /, *, axis=None, keepdims=False, ord=2):
    x, = _promote(x, atleast=float)
    special_cases = {0: xp.count_nonzero,
                     xp.inf: xp.max,
                     -xp.inf: xp.min}
    if ord in special_cases:
        return special_cases[ord](x, axis=axis, keepdims=keepdims)
    return xp.sum(abs(x)**ord, axis=axis, keepdims=keepdims)**(mp.one/ord)


# generate rough documentation
_preface = ["The following is the documentation for the corresponding "
            "attribute of `numpy.linalg`.",
            "MPArray behavior is the same except that the calculation is "
            "carried out in the appropriate precision.\n\n"]
_preface = "\n".join(_preface)
function_names = list(mod.keys())
for function_name in function_names:
    if (function_name in _imports
            or function_name[0] == '_'
            or not hasattr(np.linalg, function_name)):
        continue

    linalg_doc = getattr(np.linalg, function_name).__doc__
    mod[function_name].__doc__ = _preface + linalg_doc

    np_attr = getattr(np.linalg, function_name)
    mod_attr = mod.get(function_name)

    try:
        mod_attr.__signature__ = inspect.signature(np_attr)
    except (ValueError, TypeError):
        pass

    try:
        mod_attr.__name__ = np_attr.__name__
    except (AttributeError, TypeError):
        pass
