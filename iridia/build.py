from .main import *

from typing import Union
from math import ceil
from chgnet.model import StructOptimizer

def buildNumber(
        filename: str,
        numAtoms: int = 1000,
        format: str = None,
        **kwargs,
    ) -> tuple[Structure, Atoms]:
    atoms: Atoms = ase_read(filename, format = format)
    rnum: int = ceil(np.cbrt( numAtoms / len(atoms) ))

    rlx: Atoms = relax(atoms, **kwargs)[1]
    ratoms: Atoms = rlx * rnum

    return relax(ratoms, **kwargs)


def buildArray(
        filename: str,
        repeat: int = 1,
        format: str = None,
        **kwargs,
    ) -> tuple[Structure, Atoms]:
    atoms: Atoms = ase_read(filename, format = format)
    rlx: Atoms = relax(atoms, **kwargs)[1]
    ratoms: Atoms = rlx * repeat
    return relax(ratoms, **kwargs)


def relax(
        atoms: Union[Atoms, Structure],
        opt: str = "LBFGS",
        fmax: float = 0.0025,
        steps: int = 500,
        relax_cell = False,
        **kwargs,
    ) -> tuple[Structure, Atoms]:

    # Relax
    relaxer: StructOptimizer = StructOptimizer(optimizer_class = opt)
    result: dict[str] = relaxer.relax(atoms, fmax = fmax, steps = steps, relax_cell = relax_cell, **kwargs)

    del relaxer
    return result.get("final_structure"), result.get("trajectory").atoms

