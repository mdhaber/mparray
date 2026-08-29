"""Array API compatible, arbitrary-precision arrays."""
from importlib.metadata import version as _get_version
__version__ = _get_version("mparray")
del _get_version

from ._mparray import *
from ._mparray import __array_api_version__, __array_namespace_info__
from . import special
bool = np.bool
