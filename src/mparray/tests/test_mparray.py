import pytest
import numpy as np
import mparray as xp
from mpmath import mp


@pytest.mark.parametrize('data, type_, dtype', [
    (True, xp.bool, xp.bool),
    (1, int, xp.int64),
    (1., mp.mpf, xp.float64),
    (1+1j, mp.mpc, xp.complex128)
])
def test_default_types_dtypes(data, type_, dtype):
    x = xp.asarray(data)
    assert type(x._data[()]) == type_
    assert x.dtype == dtype


## Todo: revive this test
# def test_mparray_preserve_precision():
#     x = int(np.iinfo(np.uint64).max) + 1
#     assert xp.asarray([x])[0] == x
#
#     x = mp.mpf(np.finfo(np.float64).max) + 1
#     assert xp.asarray([x])[0] == x
#
#     x = mp.mpc(np.finfo(np.float64).max + 0j) + 1
#     assert xp.asarray([x])[0] == x
