from .main import *


def main():
    atoms = ase_read("wollastonite.cif")
    q: np.ndarray[float] = np.array([np.ones(60)]).T
    atoms.get_charges = lambda: q
    pqeqEnergy(atoms)
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


def C(rnorm: float, ai: str, aj: str = None, cutoff: float = 10., n: int = 0) -> float:
    """ Energy Between Gaussian Charges """
    """ Can be Vectorized """
    # Default Behavior
    if (aj is None):
        aj = ai

    # Alpha Coefficients
    aik: np.ndarray = np.stack([ alpha(ai, n) ], axis = 1)
    #print(f"{aik.shape = }")
    ajl: np.ndarray = np.stack([ alpha(aj, n) ], axis = 0)
    alph: np.ndarray = np.tril( (aik * ajl) / (aik + ajl) )
    ralph: np.ndarray = np.sqrt(alph)

    # Energy Array
    eGaussCharges: np.ndarray = erf(ralph * rnorm) / rnorm
    np.nan_to_num(eGaussCharges, 0.)
    np.putmask(eGaussCharges, np.isclose(rnorm, 0.), 2. * ralph / np.sqrt(np.pi))
    #eGaussCharges * Tap(rnorm, cutoff)
    #np.putmask(eGaussCharges, rnorm > cutoff, 0.)
    np.fill_diagonal(eGaussCharges, 0.)

    return eGaussCharges + eGaussCharges.T


def PQEqEnergy(positions: np.ndarray[float], spositions: np.ndarray[float], elem: list[str], charges: np.ndarray[float], cell: Cell = None, pbc: list[bool] = None, n: int = 0) -> float:
    """ Energy of Gaussian Charge Set in PQEq (Backend) """

    # Shorthand
    pos: np.ndarray = np.array(positions)
    spos: np.ndarray = np.array(spositions)
    
    if type(elem) == str:
        elem: list[str] = [elem]

    # PQEq Parameters
    params: dict[str,str: float] = loadParams(n)
    Xo: np.ndarray = np.array([ params["Xo", e] for e in elem ])
    Jo: np.ndarray = np.array([ params["Jo", e] for e in elem ])
    Ks: np.ndarray = np.array([ params["Ks", e] for e in elem ])
    Z: np.ndarray = np.array([ [params["Z", e]] for e in elem ])
    
    # Core-Shell Charge Factor
    q: np.ndarray = charges + Z

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
    Eicjc = C(ricjc, elem, n = n) * q*q.T
    Eicjs = C(ricjs, elem, n = n) * q*Z.T
    Eisjc = C(risjc, elem, n = n) * Z*q.T
    Eisjs = C(risjs, elem, n = n) * Z*Z.T

    totalEnergy = np.sum(XoEnergy + JoEnergy + springEnergy) + np.sum( Eicjc - Eicjs - Eisjc + Eisjs )
    return totalEnergy


if __name__ == "__main__":
    main()

