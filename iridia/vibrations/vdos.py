from .main import *
from .hess import *

from collections.abc import Callable

def dynamical(
        atoms: Atoms,
        verbose: str = "Solving Dynamical Matrix",
        hfunc: Callable[[Atoms], np.ndarray] = hessian,
        **kwargs,
    ) -> np.ndarray:
    """ Get the Dynamical (Hessian * 1/sqrt(m1*m2)) """
    """ for a given set of atoms. """

    mtensor: np.ndarray = np.array([ (m, m, m) for m in atoms.get_masses() ]).reshape((atoms.positions.size, 1))
    mmask: np.ndarray = 1. / np.sqrt(mtensor @ mtensor.T)
    H: np.ndarray = hfunc(atoms, **kwargs)

    return H * mmask


def vdos(atoms: Atoms, conv: float = 15.63, h: float = 1e-5) -> (np.ndarray, np.ndarray):
    """ Calculate the frequencies and vibrations of the """
    """ phonons pathways for a given set of atoms. """
    
    # Diagonalization
    dyn: np.ndarray = dynamical(atoms, h)

    return vdosDyn(dyn)


def vdosDyn(dyn: np.ndarray, conv: float = 15.63) -> (np.ndarray, np.ndarray):
    """ Calculate the Vibrational Density of States """
    """ for a given dynamical matrix. """

    # Diagonalization
    freqk, vibrations = np.linalg.eigh(dyn)
    freqk: np.ndarray = np.sqrt(freqk) * conv
    vibrations: np.ndarray = vibrations.T.reshape( len(freqk), len(freqk) // 3, 3 )

    # Remove nan
    np.nan_to_num(freqk, 0.)
    np.nan_to_num(vibrations, 0.)

    return freqk, vibrations

