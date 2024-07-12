import numpy as np
import matplotlib.pyplot as plt

from pymatgen.io.ase import MSONAtoms

from ase import Atoms
from ase.io import read as ase_read
from ase.cell import Cell
from ase.geometry import get_distances

from chgnet.model import CHGNet, CHGNetCalculator

from time import perf_counter
from tqdm import tqdm, trange
enum = lambda x, d = "" : enumerate(tqdm(x, d))

from .. import root, rroot, aroot, proot
from ..build import *


def extend(v: np.ndarray, dim: int) -> np.ndarray:
    v = np.array(v)
    for i in range(dim - v.ndim):
        v = np.stack([v], axis = -1)
    return v

