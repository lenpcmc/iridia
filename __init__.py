import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as ctime

from tqdm import tqdm
enum = lambda x: tqdm(enumerate(x))

from pymatgen.io.ase import MSONAtoms
from ase import Atoms
from ase.io import read as ase_read

from os.path import dirname, abspath
root: str = dirname(abspath(__file__))

rroot: str = f"{root}/resources"
aroot: str = f"{rroot}/atoms"
proot: str = f"{rroot}/params"

from .build import *
