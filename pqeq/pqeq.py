from .main import *

from ase.calculators.calculator import Calculator, all_changes, all_properties

from typing import Callable
from numpy.typing import ArrayLike

#from scipy.special import erf



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

    def __init__(self, atoms: Atoms, rendition: int = 0, **kwargs) -> None:
        
        super().__init__(command, **kwargs)
        
        self.shells = shellPositions(atoms.positions, elem = atoms.get_chemical_symbols(), cell = atoms.cell, pbc = atoms.pbc)
        self.charges = PQEq(atoms.positions)
        self.n = rendition
        return
    
    def calculate(self, atoms: Atoms, properties: list = implemented_properties):
        
        self.pos = atoms.positions()
        self.cell = atoms.cell.copy()
        self.pbc = atoms.pbc.copy()

        if ("energy" in properties):
            PQEqEnergy(pos, spos, symbols, charges, cell, pbc, n)

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


def alpha(elem: str, n: int = 0) -> float:
    """ Alpha Coefficient for Gaussian Charge Distribution """
    """ Can be Vectorized """
    params = loadParams(n)
    elem: list[str] = [elem] if type(elem) == str else elem
    Rk: np.ndarray = np.array([ params["Rs", e] for e in elem ])
    lambda_pqeq: float = 0.462770
    return 0.5 * lambda_pqeq / Rk**2



if __name__ == "__main__":
    main()

