from .main import *
from .hess import *

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

