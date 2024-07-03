from __init__ import *
from energy import *


def relaxShells(atoms: Atoms, n: int = 0) -> np.ndarray[float]:
    """ Relax Drude-Oscillator Shells """
    # Extract Parameters
    pos: np.ndarray = atoms.positions
    spos: np.ndarray = atoms.positions
    
    elem: list[str] = atoms.get_chemical_symbols()
    charges: np.ndarray = atoms.get_charges()
    
    cell: Cell = atoms.cell
    pbc: list[bool] = atoms.pbc

    t0 = perf_counter()
    s = shellPositions(pos, spos, elem, charges, cell, pbc, n)
    t1 = perf_counter()
    print(f"shellPos: {t1-t0}")
    return s


def dC(rvec: np.ndarray[float], ai: str, aj: str = None, n: int = 0) -> np.ndarray[float]:
    """ Force Between Gaussian Charges """
    """ Can be Vectorized """

    # Default Behavior
    if (aj == None):
        aj: str = ai

    # Alpha Coefficients
    aik: np.ndarray = np.stack([ alpha(ai, n) ], axis = 0)
    ajl: np.ndarray = np.stack([ alpha(aj, n) ], axis = 1)
    alph: np.ndarray = np.stack([np.tril( (aik * ajl) / (aik + ajl) )], axis = -1)
    ralph: np.ndarray = np.sqrt(alph)

    # Distances
    rnorm: np.ndarray = np.linalg.norm(rvec, axis = -1, keepdims = True)
    rhat: np.ndarray = rvec / rnorm

    # Derivatives
    urf: np.ndarray = erf( ralph * rnorm )
    urfp: np.ndarray = 2 * ralph * np.exp( -1. * alph * rnorm**2 ) / np.sqrt(np.pi)

    # Force Array
    fGaussCharges: np.ndarray = -1. * rhat * (urfp * rnorm - urf * 1.) / rnorm**2
    np.nan_to_num(fGaussCharges, 0.)

    for i in range(rvec.shape[-1]):
        fGaussCharges[...,i] += fGaussCharges[...,i].T
        np.putmask(fGaussCharges[...,i], np.isclose(rvec[...,i], 0.), 0.)
        np.fill_diagonal(fGaussCharges[...,i], 0.)

    return fGaussCharges


def shellPositions(positions: np.ndarray[float], spositions: np.ndarray[float] = None, elem: list[str] = None, charges: np.ndarray[float] = None, cell: Cell = None, pbc: list[bool] = None, n: int = 0) -> np.ndarray[float]:
    """ Relaxation of Shells Given a Positional State (backend) """
    
    # Default Behavior
    if (np.sum(spositions) == None):
        spositions: np.ndarray = positions
    if (elem == None):
        elem: list[str] = list( 'H' for i in range(len(positions)) )
    if (np.sum(charges) == None):
        charges: np.ndarray = np.zeros(len(positions))

    # Shorthand
    pos: np.ndarray = np.array(positions)
    spos: np.ndarray = np.array(spositions)

    # PQEq Parameters
    params: dict[str,str: float] = loadParams(n)
    Xo: np.ndarray = np.array([ params["Xo", e] for e in elem ])
    Ks: np.ndarray = np.array([ params["Ks", e] for e in elem ])
    
    # Core-Shell Charges
    Z: np.ndarray = np.array([ [params["Z", e]] for e in elem ])
    q: np.ndarray = np.array([ charges ]).T + Z

    # Core-Shell Forces
    t0 = perf_counter()
    risjs: np.ndarray = get_distances(spos, spos, cell, pbc)[0]
    risjc: np.ndarray = get_distances(spos, pos, cell, pbc)[0]
    rs: np.ndarray = np.diag( np.linalg.norm(risjc, axis = -1) )
    t1 = perf_counter()
    print(f"rij: {t1-t0}")

    t0 = perf_counter()
    Fisjs: np.ndarray = dC(risjs, elem, n = n) * np.stack([ Z*Z.T ], axis = -1)
    t1 = perf_counter()
    print(f"Fisjs: {t1-t0}")
    t0 = perf_counter()
    Fisjc: np.ndarray = dC(risjc, elem, n = n) * np.stack([ Z*q.T ], axis = -1)
    t1 = perf_counter()
    print(f"Fisjc: {t1-t0}")

    # Newton's Method
    t0 = perf_counter()
    Fext: np.ndarray = np.sum( Fisjs - Fisjc, axis = 1 )
    Fint: np.ndarray = np.array([ -1. * Ks * rs ]).T
    K: np.ndarray = np.array([ 0.5 * Ks ]).T

    spos: np.ndarray = spos + (Fext + Fint) / K
    return spos



if __name__ == "__main__":
    main()

