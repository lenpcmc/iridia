from .main import *
from .dipoles import *

def dipoleSpectrumAtoms(
        atoms: Atoms,
        vibrations: np.ndarray,
        verbose: str = "Computing Vibrational Dipole Partials",
        **kwargs,
    ) -> np.ndarray:
    
    if bool(verbose):
        dspect: np.ndarray = np.array([ atomDipolePartial( atoms, v, **kwargs ) for i,v in enum(vibrations, verbose) ])

    else:
        dspect: np.ndarray = np.array([ atomDipolePartial( atoms, v, **kwargs ) for i,v in enumerate(vibrations) ])

    return dspect


def dipoleSpectrum(
        positions: np.ndarray,
        spectrumDeltas: np.ndarray,
        charge: Callable[np.ndarray],
        verbose: str = "Computing Vibrational Dipole Partials",
        **kwargs,
    ) -> np.ndarray:

    if bool(verbose):
        dspect: np.ndarray = np.array([ dipolePartial(positions, ds, charge, **kwargs) for i,ds in enum(spectrumDeltas, verbose) ])

    else:
        dspect: np.ndarray = np.array([ dipolePartial(positions, ds, charge, **kwargs) for i,ds in enumerate(spectrumDeltas) ])

    return dspect

