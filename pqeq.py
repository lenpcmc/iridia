from __init__ import *

from ase.geometry import get_distances

from typing import Callable
from numpy.typing import ArrayLike

from scipy.special import erf

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes, all_properties
from ase.cell import Cell

class PQEqCalculator(Calculator):
    """ PQEq Calculator for ASE """
    implemented_properties = ( "energy", "forces", "dipole", "charges" )

    def __init__(self, atoms: Atoms, rendition: int = 0) -> None:
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

calculator = PQEqCalculator(None)
calculator.calculate(None)

def loadParams(n: int = 1):
    # Load Data
    with open(f"resources/params/PQEqParams{n}.csv") as infile:
        indata: list[str] = [ line.strip().split(',') for line in infile if '#' not in line ]

    # Partition
    par: list[str] = indata.pop(0)[1:]
    elements: list[str] = [ entry[0] for entry in indata ]
    atomParams: np.ndarray = np.array([ entry[1:] for entry in indata ], dtype = np.float_)
    
    # Format
    params: dict[str,str: float] = { (p,e): atomParams[i,j] for j,p in enumerate(par) for i,e in enumerate(elements) }
    return params

def alpha(elem: str, n: int = 0):
    """ Alpha Coefficient for Gaussian Charge Distribution """
    params = loadParams(n)
    elem: list[str] = [elem] if type(elem) == str else elem
    Rk: np.ndarray = np.array([ params["Rs", e] for e in elem ])
    lambda_pqeq: float = 0.462770
    return 0.5 * lambda_pqeq / Rk**2


def C(r: np.ndarray, ai: list[str], aj: list[str]):
    # Alpha Coefficients
    aik: np.ndarray = np.stack([ alpha(ai) ], axis = 1)
    ajl: np.ndarray = np.stack([ alpha(aj) ], axis = 0)
    alph: np.ndarray = (aik * ajl) / (aik + ajl)
    ralph: np.ndarray = np.sqrt(alph)
    print(ralph.shape)

    # Energy Array
    print(f"{erf(ralph * r) = }")
    print(f"{erf(ralph * r)/r = }")
    eGaussCharges: np.ndarray = erf(ralph * r) / r
    eGaussCharges[ np.isclose(r, 0.) ] = 2. * ralph / np.sqrt( np.pi )
    np.nan_to_num(eGaussCharges, 0.)
    np.fill_diagonal(eGaussCharges, 0.)

    return eGaussCharges

atoms = ase_read("wollastonite.cif")
r = get_distances(atoms.positions)
E = C(r, atoms.get_chemical_symbols(), atoms.get_chemical_symbols())
print(E)

def pqeqEnergy(
    charges: np.ndarray[float],
   positions: np.ndarray[float],
    spositions: np.ndarray[float],
    elem: list[str],
    cell: Cell = None,
    pbc: list[bool] = None,
    n: int = 1
    ):
    """ Energy of a set of atoms in PQEq """

    # Shorthand
    q: np.ndarray[float] = np.array(charges)
    pos: np.ndarray[float] = np.array(positions)
    spos: np.ndarray[float] = np.array(spositions)
    elem: list[str] = list(elem)

    # PQEq Parameters
    params: dict[str,str: float] = loadParams(n)
    Xo: np.ndarray[float] = np.array([ params["Xo", e] for e in elem ])
    Jo: np.ndarray[float] = np.array([ params["Jo", e] for e in elem ])
    Ks: np.ndarray[float] = np.array([ params["Ks", e] for e in elem ])

    # Atomic Distances
    ricjc: np.ndarray = get_distances(positions, positions, cell, pbc)[1]
    ricjs: np.ndarray = get_distances(positions, spositions, cell, pbc)[1]
    risjc: np.ndarray = get_distances(spositions, positions, cell, pbc)[1]
    risjs: np.ndarray = get_distances(positions, spositions, cell, pbc)[1]

    # Intrinsic Energy
    XoEnergy: np.ndarray = Xo * q
    JoEnergy: np.ndarray = Jo * q**2 / 2
    sprintEnergy: np.ndarray = Ks * np.diag(ricjs)**2 / 2

    # Extrinsic Energy
    Eicjc = C()
    Eicjs = C()
    Eisjc = C()
    Eisjs = C()




