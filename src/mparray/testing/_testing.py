import numpy as np
import mparray as xp


def assert_allclose(res, ref, *args, strict=True, **kwargs):
    if isinstance(res, xp.MPArray):
        res = np.asarray(res._data, dtype=res.dtype)
    return np.testing.assert_allclose(res, ref, *args, strict, **kwargs)


def assert_equal(res, ref, *args, strict=True, **kwargs):
    if isinstance(res, xp.MPArray):
        res = np.asarray(res._data, dtype=res.dtype)
    return np.testing.assert_equal(res, ref, *args, strict, **kwargs)
