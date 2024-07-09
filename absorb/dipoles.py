from .main import *
from collections.abc import Callable

def atomDipoles(atoms: Atoms) -> np.ndarray:
    return dipole( atoms.positions, atoms.get_charges() )


def dipole(pos: np.ndarray, charge: float) -> np.ndarray:
    pos: np.ndarray = np.atleast_2d(pos)
    charge: np.ndarray = np.atleast_2d(charge).T
    return pos * charge


def atomDipolePartial(atoms: Atoms, delta: np.ndarray, method: str = "central", h: float = 1e-5) -> np.ndarray:

    # Init
    patoms: Atoms = atoms.copy()
    pos: np.ndarray = atoms.positions.copy()

    # Forward Finite Method
    if (method == "forward"):
        patoms.positions = pos + delta * h
        dp: np.ndarray = atomDipoles(atoms)

        patoms.positions = pos
        dn: np.ndarray = atomDipoles(atoms)
        
        D: np.ndarray = (dp - dn) / h

    # Backward Finite Method
    elif (method == "backward"):
        patoms.positions = pos
        dp: np.ndarray = atomDipoles(atoms)

        patoms.positions = pos - delta * h
        dn: np.ndarray = atomDipoles(atoms)
        
        D: np.ndarray = (dp - dn) / h

    # Central Finite Method
    else:
        patoms.positions = pos + delta * h
        dp: np.ndarray = atomDipoles(atoms)

        patoms.positions = pos - delta * h
        dn: np.ndarray = atomDipoles(atoms)
        
        D: np.ndarray = (dp - dn) / (h * 2.)

    return D


def dipolePartial(positions: np.ndarray, delta: np.ndarray, charge: Callable[np.ndarray], method: str = "central", h: float = 1e-5) -> np.ndarray:

    # Init
    pos: np.ndarray = positions.copy()

    # Forward Finite Method
    if (method == "forward"):
        dp: np.ndarray = dipole(pos + delta * h, charge(pos + delta * h))
        dn: np.ndarray = dipole(pos, charge(pos))
        D: np.ndarray = (dp - dn) / h

    # Backward Finite Method
    elif (method == "backward"):
        dp: np.ndarray = dipole(pos, charge(pos))
        dn: np.ndarray = dipole(pos - delta * h, charge(pos + delta * h))
        D: np.ndarray = (dp - dn) / h

    # Central Finite Method
    else:
        dp: np.ndarray = dipole(pos + delta * h, charge(pos))
        dn: np.ndarray = dipole(pos - delta * h, charge(pos))
        D: np.ndarray = (dp - dn) / (h * 2.)

    return D


    return

