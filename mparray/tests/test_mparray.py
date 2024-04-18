import pytest
import numpy as np
import mparray as mp
from mparray import assert_allclose


nptypes = [# np.int8, np.int16, np.int32, np.int64,
           np.float16, np.float32, np.float64,
           np.complex64, np.complex128]


def test_mptype():
    assert mp.mptype(1) == int
    assert mp.mptype(1.) == mp.mpf
    assert mp.mptype(1+1j) == mp.mpc
    assert mp.mptype(int(1)) == int
    assert mp.mptype(mp.mpf(1)) == mp.mpf
    assert mp.mptype(mp.mpc(1)) == mp.mpc
    assert mp.mptype(mp.mparray([1])) == int
    assert mp.mptype(mp.mparray([1.])) == mp.mpf
    assert mp.mptype(mp.mparray([1.+0j])) == mp.mpc


def test_nptype():
    assert mp.nptype(1) == np.int64
    assert mp.nptype(1.) == np.float64
    assert mp.nptype(1+1j) == np.complex128
    assert mp.nptype(int(1)) == np.int64
    assert mp.nptype(mp.mpf(1)) == np.float64
    assert mp.nptype(mp.mpc(1)) == np.complex128
    assert mp.nptype(mp.mparray([1])) == np.int64
    assert mp.nptype(mp.mparray([1.])) == np.float64
    assert mp.nptype(mp.mparray([1.+0j])) == np.complex128


@pytest.mark.parametrize('nptype', nptypes)
def test_assert_allclose(nptype):
    other_dtype = np.float64 if np.issubdtype(nptype, np.integer) else np.int64
    res = [nptype(1), 2, 3]

    ref = np.asarray([1, 2, 4], dtype=nptype)
    message = "Not equal to tolerance"
    with pytest.raises(AssertionError, match=message):
        assert_allclose(mp.mparray(res), ref)

    ref = np.asarray(res, dtype=other_dtype)
    message = "`res` mptype does not match `ref` mptype."
    with pytest.raises(AssertionError, match=message):
        assert_allclose(mp.mparray(res), ref)

    ref = mp.mparray(res.copy())
    ref[1] = other_dtype(ref[1].real)
    message = "`res` mptype does not match `ref` mptype."
    with pytest.raises(AssertionError, match=message):
        assert_allclose(mp.mparray(res), ref)

    ref = np.asarray(res, dtype=nptype)
    message = "`res.shape` != `ref.shape`."
    with pytest.raises(AssertionError, match=message):
        assert_allclose(mp.mparray(res)[0:1], ref[0])

    ref = np.asarray(res, dtype=nptype)
    message = "`res` is not an mparray."
    with pytest.raises(AssertionError, match=message):
        assert_allclose(list(mp.mparray(res)), ref)

    res = mp.mparray(res)
    res[0] = nptype(res[0])
    ref = np.asarray(res, dtype=nptype)
    message = "`res` dtype is not an mptype."
    with pytest.raises(AssertionError, match=message):
        assert_allclose(res, ref)

    ref = np.asarray([0, 2, 3], dtype=nptype)
    res = mp.mparray(ref)
    message = "`res` contains trivial values."
    assert_allclose(res, ref)
    with pytest.raises(AssertionError, match=message):
        assert_allclose(res, ref, check_trivial=True)


@pytest.mark.parametrize('nptype', nptypes)
def test_mparray(nptype):
    other_dtype = np.float64 if np.issubdtype(nptype, np.integer) else np.int64

    res = [nptype(1), 2, 3]
    ref = np.asarray(res, dtype=nptype)
    assert_allclose(mp.mparray(res), ref)
    assert_allclose(mp.mparray(ref), ref)

    res = [other_dtype(1), nptype(2), nptype(3)]
    message = "`res` mptype does not match `ref` mptype."
    with pytest.raises(AssertionError, match=message):
        assert_allclose(mp.mparray(res), ref)


def test_mparray_preserve_precision():
    x = int(np.iinfo(np.uint64).max) + 1
    assert mp.asarray([x])[0] == x

    x = mp.mpf(np.finfo(np.float64).max) + 1
    assert mp.mparray([x])[0] == x

    x = mp.mpc(np.finfo(np.float64).max + 0j) + 1
    assert mp.mparray([x])[0] == x
