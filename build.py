import numpy as np
import matplotlib.pyplot as plt

from pymatgen.io.ase import MSONAtoms
from pymatgen.core import Structure
from ase import Atoms
from ase.io import read as ase_read

from chgnet.model import CHGNet, StructOptimizer

def main():
    atoms = buildArray("betaCristobalite.cif", [20,20,20])
    view(atoms)
    return


def buildNumber(filename: str, numAtoms: int = 1000, fmax: float = 0.01, steps: int = 2500) -> Atoms:
    #chgnet = CHGNet.load()
    structure = Structure.from_file(filename)
    relaxed = relaxStruct(structure, fmax, steps)
    relaxedStruct = relaxed["final_structure"]
    relaxedAtoms = relaxed["trajectory"].atoms

    repeatNumber = int(np.cbrt(numAtoms/len(relaxedStruct))) + 1
    repeatStruct = relaxedStruct.make_supercell(repeatNumber)
    repeatAtoms = relaxedAtoms * repeatNumber
    return repeatStruct, repeatAtoms


def buildArray(filename: str, repeat: int = 1, fmax: float = 0.01, steps: int = 2500, optimizer = "BFGS") -> Atoms:
    #chgnet = CHGNet.load()
    structure = Structure.from_file(filename)
    relaxed = relaxStruct(structure, fmax, steps)
    relaxedAtoms = relaxed["trajectory"].atoms
    relaxedStruct = relaxed["final_structure"]
    repeatAtoms = relaxedAtoms * repeat
    repeatStruct = relaxedStruct.make_supercell(repeat)
    return repeatStruct, repeatAtoms


def relaxStruct(structure: Structure, fmax: float = 0.01, steps: int = 2500, save = False) -> dict[str]:
    relaxer = StructOptimizer()
    result = relaxer.relax(structure, fmax = fmax, steps = steps)
    
    #print(f"Relaxed structure {result['final_structure']}")
    #print(f"Relaxed total energy {result['trajectory'].energies[-1]} eV")

    if bool(save):
        pass

    return result


if __name__ == "__main__":
    main()
