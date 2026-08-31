"""Array API compatible, arbitrary-precision arrays."""
from importlib.metadata import version as _get_version

__version__ = _get_version("mparray")
del _get_version

from numpy import bool

from mparray import special
from mparray._mparray import *
from mparray._mparray import __array_api_version__, __array_namespace_info__
