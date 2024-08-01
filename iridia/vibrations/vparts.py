from .main import *
from ase import Atom
from collections.abc import Callable

def vparts(vibrations: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vibrations, axis = 2)


def categorize(
        atoms: Atoms,
        choose: Callable[Atom] = lambda a: a.number,
    ) -> np.ndarray[bool]:
    
    assign: list = [ choose(a) for a in atoms ]
    cmask: np.ndarray = np.array([ [ assign[i] == cat for i,a in enumerate(assign) ] for cat in np.unique(assign) ])

    return cmask


def categoryParts(
        atoms: Atoms,
        vibrations: np.ndarray,
        choose: Callable[Atom] = lambda a: a.number,
    ) -> np.ndarray:

    parts: np.ndarray = extend(vparts(vibrations), 3)
    cats: np.ndarray = extend(categorize(atoms, choose), 3).swapaxes(0,2)
    cparts: np.ndarray = parts * cats

    return np.sum(cparts**2., axis = 1, keepdims = True)

