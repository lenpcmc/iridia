from .main import *

from scipy.special import erf
from numpy import broadcast_to as broadcast

def main():
    #from charge import pqeq
    #atoms = ase_read("wollastonite.sdf")
    #structure, atoms = buildArray("wollastonite.cif", [1,1,1])
    structure, atoms = buildArray("betaCristobalite.cif", [1,1,1])
    atoms.get_charges = lambda: atoms.arrays.get("oxi_states")
    for i in range(1):
        q = pqeq(atoms)
        rs = relaxShells(atoms)
        plt.scatter(rs[...,0], rs[...,1])
    #print(f"{q = }")
    print("\n".join([ f"{atoms.get_chemical_symbols()[i]}, {q[i]}" for i in range(len(q)) ]))
    plt.show()
    return


def relaxShells(atoms: Atoms, n: int = 0) -> np.ndarray[float]:
    """ Relax Drude-Oscillator Shells """
    # Extract Parameters
    pos: np.ndarray = atoms.positions
    spos: np.ndarray = atoms.positions
    
    elem: list[str] = atoms.get_chemical_symbols()
    
    cell: Cell = atoms.cell
    pbc: list[bool] = atoms.pbc

    # Charge Conditional
    try:
        charges: np.ndarray = atoms.get_charges()
    except RuntimeError:
        charges: np.ndarray = np.zeros(len(atoms))

    return shellPositions(pos, spos, elem, charges, cell, pbc, n)


def dC(rvec: np.ndarray[float], ai: str, aj: str = None, cutoff: float = 10., n: int = 0) -> np.ndarray[float]:
    """ Force Between Gaussian Charges """
    """ Can be Vectorized """

    # Default Behavior
    if (aj is None):
        aj: str = ai

    # Alpha Coefficients
    aik: np.ndarray = np.stack([ alpha(ai, n) ], axis = 1)
    ajl: np.ndarray = np.stack([ alpha(aj, n) ], axis = 0)
    alph: np.ndarray = np.stack([np.tril( (aik * ajl) / (aik + ajl) )], axis = -1)
    ralph: np.ndarray = np.sqrt(alph)

    # Distances
    rnorm: np.ndarray = broadcast( np.linalg.norm(rvec, axis = -1, keepdims = True), rvec.shape )
    rhat: np.ndarray = rvec / rnorm

    # Derivatives
    urf: np.ndarray = erf( ralph * rnorm )
    urfp: np.ndarray = 2. * ralph * np.exp( -1. * alph * rnorm**2. ) / np.sqrt(np.pi)

    # Force Array
    fGaussCharges: np.ndarray = -1. * rhat * (urfp * rnorm - urf * 1.) / rnorm**2.
    np.nan_to_num(fGaussCharges, 0.)
    np.putmask(fGaussCharges, np.isclose(rnorm, 0.), 0.)
    #np.putmask(fGaussCharges, np.broadcast_to(rnorm > cutoff, fGaussCharges.shape), 0.)
    #fGaussCharges * Tap(rvec, cutoff)

    fGaussCharges += -1. * fGaussCharges.swapaxes(0,1)

    return fGaussCharges


def shellPositions(positions: np.ndarray[float], spositions: np.ndarray[float] = None, elem: list[str] = None, charges: np.ndarray[float] = None, cell: Cell = None, pbc: list[bool] = None, n: int = 0) -> np.ndarray[float]:
    """ Relaxation of Shells Given a Positional State (backend) """
    
    # Default Behavior
    if (spositions is None):
        spositions: np.ndarray = positions
    if (elem is None):
        elem: list[str] = list( 'H' for i in range(len(positions)) )
    if (charges is None):
        charges: np.ndarray = np.zeros(len(positions))

    # Shorthand
    pos: np.ndarray = np.array(positions)
    spos: np.ndarray = np.array(spositions)

    # PQEq Parameters
    params: dict[str,str: float] = loadParams(n)
    Ks: np.ndarray = np.array([ [params["Ks", e]] for e in elem ])
    
    # Core-Shell Charges
    Z: np.ndarray = np.array([ [params["Z", e]] for e in elem ])
    q: np.ndarray = np.array([ charges ]).T + Z

    # Core-Shell Forces
    risjs: np.ndarray = get_distances(spos, spos, cell, pbc)[0]
    risjc: np.ndarray = get_distances(spos, pos, cell, pbc)[0]

    Fisjs: np.ndarray = -14.4 * dC(risjs, elem, n = n) * np.stack([ Z*Z.T ], axis = -1)
    Fisjc: np.ndarray = -14.4 * dC(risjc, elem, n = n) * np.stack([ Z*q.T ], axis = -1)

    # Intra Atomic Interactions
    idx = np.arange(risjc.shape[0])
    rsv: np.ndarray = risjc[idx, idx, :]

    # Newton's Method
    Fext: np.ndarray = np.sum( Fisjs - Fisjc, axis = 0 )
    Fint: np.ndarray = Ks * rsv
    K: np.ndarray = Ks

    spos: np.ndarray = spos + (Fext + Fint) / K
    return spos


if __name__ == "__main__":
    main()

