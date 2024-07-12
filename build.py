from .main import *

from typing import Union
from chgnet.model import StructOptimizer

from io import StringIO
import sys

def buildNumber(filename: str, numAtoms: int = 1000, format: str = None, **kwargs) -> tuple[Structure, Atoms]:
    atoms: Atoms = ase_read(filename, format = format)
    rnum: int = int(np.cbrt( numAtoms / len(atoms) )) + 1

    ratoms: Atoms = relax(atoms, **kwargs)[1]
    rratoms: Atoms = atoms * rnum

    return relax(rratoms, **kwargs)


def buildArray(filename: str, repeat: int = 1, format: str = None, **kwargs) -> tuple[Structure, Atoms]:
    atoms: Atoms = ase_read(filename, format = format)
    ratoms: Atoms = relax(atoms, **kwargs)[1]
    rratoms: Atoms = ratoms * repeat
    return relax(rratoms, **kwargs)


def relax(atoms: Union[Atoms, Structure], opt: str = "LBFGS", fmax: float = 0.0025, steps: int = 2500, **kwargs) -> tuple[Structure, Atoms]:

    # Relax
    relaxer = StructOptimizer(optimizer_class = opt)
    result: dict[str] = relaxer.relax(atoms, fmax = fmax, steps = steps, **kwargs)

    return result.get("final_structure"), result.get("trajectory").atoms

