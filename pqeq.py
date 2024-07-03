from __init__ import *

from ase.geometry import get_distances

from typing import Callable
from numpy.typing import ArrayLike

from scipy.special import erf

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes, all_properties
from ase.cell import Cell


def main():
    atoms = ase_read("wollastonite.cif")
    q: np.ndarray[float] = np.array([np.ones(60)]).T
    r = get_distances(atoms.positions)[1]
    E = C(r, atoms.get_chemical_symbols(), atoms.get_chemical_symbols())
    print(E)
    return


class PQEqCalculator(Calculator):
    """ PQEq Calculator for ASE """
    implemented_properties = ( "energy", "forces", "dipole", "charges" )

    def __init__(self, atoms: Atoms, rendition: int = 0) -> None:
        self.atoms = atoms
        self.n = rendition
        return
    
    def calculate(self, atoms: Atoms, properties: list = implemented_properties):

        if ("energy" in properties):
            #pqeqEnergy(atoms)
            print("energy")
        if ("forces" in properties):
            #pqeqForces(atoms)
            print("forces")
        if ("dipole" in properties):
            #pqeqDipole(atoms)
            print("dipole")
        if ("charges" in properties):
            #pqeq(atoms)
            print("charges")

        return 


def loadParams(n: int = 0, eV: bool = True):
    # Load Data
    with open(f"resources/params/PQEqParams{n}.csv") as infile:
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


if __name__ == "__main__":
    main()

