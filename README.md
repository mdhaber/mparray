MPArray is an [Array API Standard](https://data-apis.org/array-api/latest/) compatible array library that features arbitrary precision arithmetic.

Install with `pip`:

```shell
pip install mparray
```

Import to access the array namespace:

```shell
import mparray as xp
```

Use any feature defined by the 2025.12 version of the standard:

```shell
x = xp.arange(5)
# MPArray([0, 1, 2, 3, 4], dtype=int64)
```

Integer arrays are filled with Python `int`s; consequently, elements never overflow.

```shell
x = x[:2] + 10**50
# MPArray([100000000000000000000000000000000000000000000000000,
#          100000000000000000000000000000000000000000000000001],
#         dtype=int64)
```

Dtypes follow array API promotion rules for compatibility, but do not limit the underlying representation. For instance, `int32` arrays also use Python `int`s.

```shell
xp.astype(x, xp.int32)
# MPArray([100000000000000000000000000000000000000000000000000,
#          100000000000000000000000000000000000000000000000001],
#         dtype=int32)
```

Real and complex floating point arrays are backed by [`mpmath`](https://github.com/mpmath/mpmath).

```shell
from mpmath import mp
mp.dps = 55  # set the desired precision
y = xp.astype(x, xp.float64)
# MPArray([mpf('100000000000000000000000000000000000000000000000000.0'),
#          mpf('100000000000000000000000000000000000000000000000001.0')],
#         dtype=float64)
xp.exp(y)
# MPArray([mpf('4.535356657536074105363661926352788592352627616652483210606e+43429448190325182765112891891660508229439700580366'),
#          mpf('1.232837758776106335937289338644047193491770865081040599056e+43429448190325182765112891891660508229439700580367')],
#         dtype=float64)
```

Arbitrary-precision equivalents of some SciPy special functions are available:

```shell
xp.special.ndtr(-y)
# MPArray([mpf('8.998129248551223738202088675072137970394711995443364311445e-2171472409516259138255644594583025411471985029018332830572230087781144236110599295949020894164292471'),
#          mpf('6.461085845597172042837958170074610008600868697345677551181e-2171472409516259138255644594583025411471985029018376260020420755455996799425617236333573042128793971')],
#         dtype=float64)
```
