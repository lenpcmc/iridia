import numpy as np
import matplotlib.pyplot as plt

from pymatgen.core import Structure
from pymatgen.io.ase import MSONAtoms

from ase import Atoms
from ase.io import read as ase_read, write as ase_write

from time import perf_counter
from tqdm import tqdm, trange

enum = lambda x, d = "" : enumerate(tqdm(x, d))

def extend(v: np.ndarray, dim: int) -> np.ndarray:
    v = np.array(v)
    v.shape += tuple(np.ones( (dim - v.ndim) * (dim > v.ndim), dtype = int ))
    return v

