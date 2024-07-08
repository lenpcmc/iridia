from os.path import dirname, abspath
root: str = dirname(abspath(__file__))

rroot: str = f"{root}/resources"
aroot: str = f"{rroot}/atoms"
proot: str = f"{rroot}/params"

from .build import *

from .pqeq.pqeq import *
from .pqeq.charge import *
from .pqeq.energy import *
from .pqeq.force import *

from .vibrations.vdos import *
from .vibrations.vplot import *
