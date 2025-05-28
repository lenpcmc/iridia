from .iridia import *

# Hax
from sys import modules
class Iridia(modules[__name__].__class__):
    def __call__(self, *args, **kwargs):
        return self.ir.iridia(*args, **kwargs)
modules[__name__].__class__ = Iridia
