from os.path import dirname, abspath
ir_root: str = dirname(abspath(__file__))

from . import ir

from . import pqeq
from . import vibrations
from . import absorb

from . import build
from . import visualize

__version__: str = "1.0.0"
