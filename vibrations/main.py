import numpy as np
import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read as ase_read
from ase.cell import Cell
from ase.geometry import get_distances

from chgnet.model import CHGNet, CHGNetCalculator

from time import perf_counter
from tqdm import tqdm, trange
enum = lambda x, d = "" : enumerate(tqdm(x, d))

from .. import ir_root
from ..build import *


def extend(v: np.ndarray, dim: int) -> np.ndarray:
    v = np.array(v)
    v.shape += tuple(np.ones( dim - v.ndim, dtype = int ))
    return v

