from os.path import dirname, abspath
ir_root: str = dirname(abspath(__file__))

__version__: str = "1.0.0"

from .ir import *

from . import pqeq
from . import vibrations
from . import absorb

from . import build
from . import visualize

# Hax
from sys import modules
class Iridia(modules[__name__].__class__):
    def __call__(self, *args, **kwargs):
        return self.ir.iridia(*args, **kwargs)
modules[__name__].__class__ = Iridia
