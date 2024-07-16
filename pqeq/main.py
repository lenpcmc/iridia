import numpy as np
import matplotlib.pyplot as plt

from pymatgen.io.ase import MSONAtoms

from ase import Atoms
from ase.io import read as ase_read
from ase.cell import Cell
from ase.geometry import get_distances

from time import perf_counter
from tqdm import tqdm

from .. import ir_root, enum
from ..build import *


def loadParams(n: int = 0, eV: bool = True):
    # Load Data
    with open(f"{ir_root}/pqeq/params/PQEqParams{n}.csv") as infile:
        indata: list[str] = [ line.strip().split(',') for line in infile if '#' not in line ]

    # Partition
    par: list[str] = indata.pop(0)[1:]
    elements: list[str] = [ entry[0] for entry in indata ]
    atomParams: np.ndarray = np.array([ entry[1:] for entry in indata ], dtype = np.float_)

    # K Conversion 
    if (eV):
        ki = par.index("Ks")
        atomParams[:,ki] *= 0.04336
    
    # Format
    params: dict[str,str: float] = { (p,e): atomParams[i,j] for j,p in enumerate(par) for i,e in enumerate(elements) }
    return params


def alpha(elem: str, n: int = 0) -> float:
    """ Alpha Coefficient for Gaussian Charge Distribution """
    """ Can be Vectorized """
    params = loadParams(n)
    elem: list[str] = [elem] if type(elem) == str else elem
    Rk: np.ndarray = np.array([ params["Rs", e] for e in elem ])
    lambda_pqeq: float = 0.462770
    return 0.5 * lambda_pqeq / Rk**2


#def Tap(r, rcut = 12.5):
#    A = [1, 0, 0, 0, -35, 84, -70, 20]
#    return np.sum([ ( -r / rcut )**a for a in A ])

