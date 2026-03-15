"""Array API compatible, arbitrary-precision arrays."""
__version__ = "0.0.0"
from ._mparray import *
from ._mparray import __array_api_version__, __array_namespace_info__
bool = np.bool
