from .main import *
from collections.abc import Callable

def vparts(vibrations: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vibrations, axis = 2)


def categorize(
        atoms: Atoms,
        choose: Callable[[Atoms], list] = lambda a: a.number,
    ) -> np.ndarray[bool]:
    
    assign: list = choose(atoms)
    cmask: np.ndarray = np.array([ [ assign[i] == cat for i,a in enumerate(assign) ] for cat in np.unique(assign) ])

    return cmask


def categoryParts(
        atoms: Atoms,
        vibrations: np.ndarray[float],
        choose: Callable[[Atoms], list] = lambda a: a.get_chemical_symbols(),
    ) -> np.ndarray[float]:

    parts: np.ndarray = extend(vparts(vibrations), 3)
    cats: np.ndarray = extend(categorize(atoms, choose), 3).swapaxes(0,2)
    cparts: np.ndarray = parts * cats

    return np.sum(cparts**2., axis = 1, keepdims = True)


def coordNumber(atoms: Atoms, rcut: float = 2.) -> np.ndarray[float]:
    dij: np.ndarray = atoms.get_all_distances(mic = True)
    return np.sum( dij <= rcut, axis = 0, ) - 1

