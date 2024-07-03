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


def pqeqEnergy(atoms: Atoms, n: int = 0) -> float:
    #
    pos: np.darray = atoms.positions
    spos: np.ndarray = relaxShells(atoms, n)

    elem: list[str] = atoms.get_chemical_symbols()
    charges: np.ndarray = atoms.get_charges()

    cell: Cell = atoms.cell
    pbc: list[bool] = atoms.pbc

    return PQEqEnergy(pos, spos, elem, charges, cell, pbc, n)


def alpha(elem: str, n: int = 0) -> float:
    """ Alpha Coefficient for Gaussian Charge Distribution """
    """ Can be Vectorized """
    params = loadParams(n)
    elem: list[str] = [elem] if type(elem) == str else elem
    Rk: np.ndarray = np.array([ params["Rs", e] for e in elem ])
    lambda_pqeq: float = 0.462770
    return 0.5 * lambda_pqeq / Rk**2


def C(r: float, ai: str, aj: str = None, n: int = 0) -> float:
    """ Energy Between Gaussian Charges """
    """ Can be Vectorized """
    # Default Behavior
    if (aj == None): aj = ai

    # Alpha Coefficients
    aik: np.ndarray = np.stack([ alpha(ai, n) ], axis = 0)
    ajl: np.ndarray = np.stack([ alpha(aj, n) ], axis = 1)
    alph: np.ndarray = np.tril( (aik * ajl) / (aik + ajl) )
    ralph: np.ndarray = np.sqrt(alph)

    # Energy Array
    eGaussCharges: np.ndarray = erf(ralph * r) / r
    np.nan_to_num(eGaussCharges, 0.)
    np.putmask(eGaussCharges, np.isclose(r, 0.), 2. * ralph / np.sqrt(np.pi))
    np.fill_diagonal(eGaussCharges, 0.)

    return eGaussCharges + eGaussCharges.T


def PQEqEnergy(positions: np.ndarray[float], spositions: np.ndarray[float], elem: list[str], charges: np.ndarray[float], cell: Cell = None, pbc: list[bool] = None, n: int = 0) -> float:
    """ Energy of Gaussian Charge Set in PQEq (Backend) """

    # Shorthand
    pos: np.ndarray = np.array(positions)
    spos: np.ndarray = np.array(spositions)
    if type(elem) == str: elem: list[str] = [elem]

    # PQEq Parameters
    params: dict[str,str: float] = loadParams(n)
    Xo: np.ndarray = np.array([ params["Xo", e] for e in elem ])
    Jo: np.ndarray = np.array([ params["Jo", e] for e in elem ])
    Ks: np.ndarray = np.array([ params["Ks", e] for e in elem ])
    Z: np.ndarray = np.array([ [params["Z", e]] for e in elem ])
    
    # Core-Shell Charge Factor
    q: np.ndarray = np.array([ charges + Z ])

    # Atomic Distances
    ricjc: np.ndarray = get_distances(positions, positions, cell, pbc)[1]
    ricjs: np.ndarray = get_distances(positions, spositions, cell, pbc)[1]
    risjc: np.ndarray = get_distances(spositions, positions, cell, pbc)[1]
    risjs: np.ndarray = get_distances(positions, spositions, cell, pbc)[1]

    # Intrinsic Energy
    XoEnergy: np.ndarray = Xo * q
    JoEnergy: np.ndarray = Jo * q**2 / 2
    springEnergy: np.ndarray = Ks * np.diag(ricjs)**2 / 2

    # Extrinsic Energy
    Eicjc = C(ricjc, elem, elem) * q*q.T
    Eicjs = C(ricjs, elem, elem) * q*Z.T
    Eisjc = C(risjc, elem, elem) * Z*q.T
    Eisjs = C(risjs, elem, elem) * Z*Z.T

    totalEnergy = np.sum(XoEnergy + JoEnergy + springEnergy) + np.sum( Eicjc - Eicjs - Eisjc + Eisjs )
    return totalEnergy


if __name__ == "__main__":
    main()

