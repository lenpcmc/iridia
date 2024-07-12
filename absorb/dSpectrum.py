from .main import *
from .dipoles import *

def dipoleSpectrumAtoms(atoms: Atoms, vibrations: np.ndarray, method: str = "central", h: float = 1e-5) -> np.ndarray:
    return np.array([ atomDipolePartial( atoms, v, method, h ) for i,v in enum(vibrations, "Computing Vibrational Dipole Partials") ])


def dipoleSpectrum(positions: np.ndarray, spectrumDeltas: np.ndarray, charge: Callable[np.ndarray], erbose = True, method: str = "central", h: float = 1e-5) -> np.ndarray:
    return np.array([ dipolePartial(positions, ds, charge, method, h) for i,ds in enum(spectrumDeltas) ])

