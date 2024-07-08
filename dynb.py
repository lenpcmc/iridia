import numpy as np
import matplotlib.pyplot as plt

from ase.atom import Atom
from chgnet.model import CHGNet, StructOptimizer, CHGNetCalculator
from pymatgen.core import Structure
from pymatgen.io.ase import MSONAtoms, AseAtomsAdaptor

from __init__ import *
from pqeq import *
#from pqeq.force import *

from tqdm import *

from build import *

def main():
    structure, atoms = buildArray("resources/atoms/betaCristobalite.cif", 3)
    Dyn = dynamical(atoms)
    plt.imshow(Dyn)
    plt.show()
    return Dyn


def dynamical(atoms :MSONAtoms, h: float = 1e-5, verbose: int = None) -> np.ndarray:
    dynamicVector = np.array([ 3 * [a.mass] for a in atoms ]).reshape(1, 3*len(atoms))
    dynamicVector = 1./np.sqrt(dynamicVector)
    dynamicScale = dynamicVector.T @ dynamicVector
    return -1. * dynamicScale * hessian(atoms, h=h)


def hessian(atoms: MSONAtoms, h: float = 1e-5) -> np.ndarray:
    pos = np.array([ atoms.positions.copy() for i in trange( 3*len(atoms) ) ])
    print(pos.shape)

    for i,p in enumerate(tqdm(pos)):
        pos[i, i // 3, i % 3] += 1e-6
        print(p.shape)
    fp = np.array([ PQEqForce(p) for p in tqdm(pos) ])
    exit()

    return (H + H.T) / 2.


def hessRow(atoms: MSONAtoms, i: int, h: float = 1e-5) -> np.ndarray:
    # Init
    atoms.calc = CHGNetCalculator() if atoms.calc == None else atoms.calc
    pos: tuple[int] = ( i // 3, i % 3 )

    # Positive Direction
    atoms.positions[pos] += h
    fp = atoms.get_forces()

    # Negative Direction
    atoms.positions[pos] -= 2. * h
    fm = atoms.get_forces()

    # Derivative
    D = (fp - fm) / (2. * h)
    return D.flatten()



if __name__ == "__main__":
    main()
