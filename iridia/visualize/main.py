import numpy as np
import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read as ase_read

from time import perf_counter
from tqdm import tqdm, trange

import scienceplots
plt.style.use(["science", "no-latex"])

from .. import ir_root
from ..main import enum, extend
from ..build import *

