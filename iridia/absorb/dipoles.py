from .main import *
from collections.abc import Callable

def atomDipoles(atoms: Atoms) -> np.ndarray:
    return dipole( atoms.positions, atoms.get_charges() )


def dipole(pos: np.ndarray, charge: float) -> np.ndarray:
    pos: np.ndarray = extend(pos, 3)
    charge: np.ndarray = extend(charge, 3).swapaxes(1,2)
    return pos * charge


def atomDipolePartial(atoms: Atoms, delta: np.ndarray, method: str = "central", h: float = 1e-5) -> np.ndarray:

    # Init
    pos: np.ndarray = atoms.positions.copy()
    patoms: Atoms = atoms.copy()
    patoms.calc = atoms.calc
    patoms.get_charges = atoms.get_charges

    # Forward Finite Method
    if (method == "forward"):
        patoms.positions = pos + delta * h
        dp: np.ndarray = atomDipoles(patoms)

        patoms.positions = pos
        dn: np.ndarray = atomDipoles(patoms)
        
        D: np.ndarray = (dp - dn) / h

    # Backward Finite Method
    elif (method == "backward"):
        patoms.positions = pos
        dp: np.ndarray = atomDipoles(patoms)

        patoms.positions = pos - delta * h
        dn: np.ndarray = atomDipoles(patoms)
        
        D: np.ndarray = (dp - dn) / h

    # Central Finite Method
    else:
        patoms.positions = pos + delta * h
        dp: np.ndarray = atomDipoles(patoms)

        patoms.positions = pos - delta * h
        dn: np.ndarray = atomDipoles(patoms)
        
        D: np.ndarray = np.sum(dp - dn, axis = 0) / (h * 2.)

    return D


def dipolePartial(
        positions: np.ndarray,
        delta: np.ndarray,
        charge: Callable[[np.ndarray], np.ndarray],
        method: str = "central",
        h: float = 1e-5,
    ) -> np.ndarray:

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
        D: np.ndarray = np.sum(dp - dn, axis = 0) / (h * 2.)

    return D

