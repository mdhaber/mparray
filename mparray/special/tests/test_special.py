import numpy as np
from mpmath import mp
from scipy import special as sps
from mparray import special as mps


def assert_allclose(res, ref):
    np.testing.assert_allclose(res.astype(np.float64), ref)


def assert_mp(res):
    res0 = res[0] if res.ndim > 0 else res[()]
    assert (isinstance(res0, mp.mpf)
            or isinstance(res0, mp.mpc)
            or isinstance(res0, mp.constant))


def test_log1p():  # parameterize over functions
    x = 0.5
    res, ref = mps.log1p(x), sps.log1p(x)
    assert_mp(res)
    assert_allclose(res, ref)