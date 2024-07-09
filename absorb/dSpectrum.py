from .main import *
from .dipoles import *
from ..vibrations.vdos import *

def dipoleSpectrumAtoms(atoms: Atoms, vibrations: np.ndarray, method: str = "central", h: float = 1e-5) -> np.ndarray:
    return np.array([ atomDipolePartial( atoms, v, method, h ) for i,v in enum(vibrations, "Computing Vibrational Dipole Partials") ])


def dipoleSpectrum(positions: np.ndarray, spectrumDeltas: np.ndarray, charge: Callable[np.ndarray], method: str = "central", h: float = 1e-5) -> np.ndarray:
    # Init
    pos: np.ndarray = np.atleast_3d( positions.copy() ).reshape( 1, len(positions), 3 )
    ds: np.ndarray = spectrumDeltas.copy()
    charge3 = lambda delta: np.array([ charge(d) for d in delta ])

    return dipolePartial(pos, ds, charge3, method, h)

