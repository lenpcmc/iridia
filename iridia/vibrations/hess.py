from .main import *

from collections.abc import Callable

def hessian(
        atoms: Atoms,
        verbose: str = "Solving Hessian Matrix",
        **kwargs,
    ) -> np.ndarray:
    """ Get the Hessian (dE/dn,dm) for a given """
    """ set of atoms. """

    # Ensure Calculator
    if (atoms.calc is None):
        atoms.calc = CHGNetCalculator()

    # Allocate and Fill
    H: np.ndarray = np.array([
        hessRow(atoms, i, **kwargs) for i in trange( 3*len(atoms), desc = f"{verbose}" )
    ])

    # Ensure Symmetry over Numeric Precision
    return -1. * (H + H.T) / 2.


def hessRow(atoms: Atoms, i: int, method: str = "central", h: float = 1e-5) -> np.ndarray:
    """ Find the ith row of the Hessian """
    """ for a given set of atoms. """

    # Shorthand
    apos: np.ndarray = atoms.positions
    
    # Finite Forward Difference
    if (method == "foward"):
        fn: np.ndarray = atoms.get_forces().flatten()
        apos[i // 3, i % 3] += h

        fp: np.ndarray = atoms.get_forces().flatten()
        apos[i // 3, i % 3] -= h
    
        D: np.ndarray = (fp - fn) / h

    elif (method == "backward"):
        fp: np.ndarray = atoms.get_forces().flatten()
        apos[i // 3, i % 3] -= h
        
        fn: np.ndarray = atoms.get_forces().flatten()
        apos[i // 3, i % 3] += h
    
        D: np.ndarray = (fp - fn) / h
    
    else:
        apos[i // 3, i % 3] += h

        fp: np.ndarray = atoms.get_forces().flatten()
        apos[i // 3, i % 3] -= h * 2.
        
        fn: np.ndarray = atoms.get_forces().flatten()
        apos[i // 3, i % 3] += h
        
        D: np.ndarray = (fp - fn) / (2. * h)

    return D

