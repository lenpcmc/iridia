import numpy as np
import matplotlib.pyplot as plt

from pymatgen.io.ase import MSONAtoms

from ase import Atoms
from ase.io import read as ase_read
from ase.cell import Cell
from ase.geometry import get_distances

from chgnet.model import CHGNet, CHGNetCalculator

from tqdm import tqdm, trange
enum = lambda x: tqdm(enumerate(x))

from time import perf_counter

#from .. import root, rroot, aroot, proot
root = "../"
rroot = f"{root}/resources"
aroot = f"{rroot}/atoms"
proot = f"{rroot}/params"
#from ..build import *
from build import *

