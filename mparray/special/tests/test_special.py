import pytest
import numpy as np
from mpmath import mp
from scipy import special as sps
from mparray import special as mps


def assert_allclose(res, ref):
    res = (np.asarray(res, dtype=np.float64) if np.all(mps.real(res) == res)
           else np.asarray(res, dtype=np.complex128))
    assert np.all(np.isfinite(res) & (res != 0) & (res != 1))
    np.testing.assert_allclose(res, ref)


def assert_mp_type(res):
    res0 = res[0] if res.ndim > 0 else res[()]
    assert (isinstance(res0, mp.mpf)
            or isinstance(res0, mp.mpc)
            or isinstance(res0, mp.constant))


# these arguments happen to work for most functions
goodargs = (3.123, 2.456, 1.789, 0.234)

intarg = {'factorial2', 'lambertw'}
arg01 = {'ndtri', 'logit', 'betainc'}


@pytest.mark.parametrize('shape', [tuple(), (2,)])
@pytest.mark.parametrize(['f_name', 'nargs'], [
    ['expm1', 1], ['log1p', 1], ['cosm1', 1], ['psi', 1], ['digamma', 1],
    ['ndtr', 1], ['gammaln', 1], ['erf', 1], ['erfc', 1], ['zeta', 1],
    ['poch', 2], ['binom', 2], ['comb', 2], ['powm1', 2], ['hyp1f1', 3],
    ['hyp2f1', 4], ['iv', 2], ['kv', 2], ['gammainc', 2], ['gammaincc', 2],
    ['log_ndtr', 1], ['betaln', 2], ['xlogy', 2], ['xlog1py', 2], ['cosm1', 1],
    ['expit', 1], ['boxcox', 2], ['boxcox1p', 2], ['ive', 2], ['i0e', 1],
    ['i1e', 1], ['kve', 2], ['k0e', 1], ['k1e', 1], ['factorial2', 1],
    ['lambertw', 2], ['logit', 1], ['ndtri', 1], ['chdtr', 2], ['chdtrc', 2],
    ['betainc', 3], ['fdtr', 3], ['fdtrc', 3], ['stdtr', 2]
])
def test_special_real(shape, f_name, nargs):
    f_mps = getattr(mps, f_name)
    f_sps = getattr(sps, f_name)
    args = list(goodargs[:nargs])

    if f_name in intarg:
        args[-1] = int(args[-1])
    if f_name in arg01:
        args = [arg/4 for arg in args]

    args = [np.broadcast_to(arg, shape)[()] for arg in args]

    res, ref = f_mps(*args), f_sps(*args)
    assert_mp_type(res)
    assert_allclose(res, ref)


@pytest.mark.parametrize('case', [
    ('log_ndtr', (-30,)), ('log_ndtr', (30,)),  # ndtr is very close to 0 or 1
    ('betaln', (1e10, 1e20)),  # beta is very small
    ('betaln', (1, 1+1e-10)),  # beta is nearly 1
    ('betaln', (1e10, 1e-20)),  # beta is very large
    ('cosm1', (1e-20,)),  # cos is very close to 1
])
def test_special_edge(case):
    # test edge cases where accuracy is suspected to be challenging
    f_name, args = case
    f_mps = getattr(mps, f_name)
    f_sps = getattr(sps, f_name)

    res, ref = f_mps(*args), f_sps(*args)
    assert_mp_type(res)
    assert_allclose(res, ref)


@pytest.mark.parametrize('axis', (0, 1))
def test_logsumexp(axis):
    # logsumexp is unusual in that it does a reduction of an array; test separately
    rng = np.random.default_rng(8583692938552)
    a = rng.random((3, 4))
    b = rng.random((3, 4))
    kwargs = dict(a=a, axis=axis, b=b)
    res = mps.logsumexp(**kwargs)
    ref = sps.logsumexp(**kwargs)
    assert_mp_type(res)
    assert_allclose(res, ref)
