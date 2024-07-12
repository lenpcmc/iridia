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
enum = lambda x: enumerate(tqdm(x))

from .. import root, rroot, aroot, proot
from ..build import *
from ..vibrations.vdos import *

def main():
    atoms = ase_read(f"{aroot}/wollastonite.cif")
    dyn = np.load(f"{rroot}/arrays/wollastonite.npy")
    freqk, vibrations = vdosDyn(dyn)
    vibration = vibrations[50]
    animateVibration(atoms, vibration)
    animateAtoms(atoms, None)
    return


def animateAtoms(atoms: Atoms, path: np.ndarray, fps: float = 30):
    
    # Get Bounds
    bounds: np.ndarray = np.array([ np.min(atoms.cell, axis = 1), np.max(atoms.cell, axis = 1) ])
    pad: np.ndarray = 0.2 * (bounds[1] - bounds[0])
    pbounds: np.ndarray = np.array([bounds[0] - pad, bounds[1] + pad])

    # Matplotlib Setup
    fig,ax = plt.subplots( subplot_kw = {"projection": "3d"} )
    plt.axis( pbounds.T.flatten() )

    def animate(frame):
        while bool(ax.collections):
            ax.collections[0].remove()
        spos: np.ndarray = atoms.positions + path[frame]
        ax.scatter( xs = spos[:,0], ys = spos[:,1], zs = spos[:,2] )
        
        ppos: np.ndarray = atoms.positions + path[:frame]


def animateVibration(atoms: Atoms, vibration: np.ndarray, frames: int = 60, fps: float = 30):
    path: np.ndarray = vibration * np.sin(2. * np.pi * np.arange(frames) / fps)
    print(f"{path.shape = }")


if __name__ == "__main__":
    main()
