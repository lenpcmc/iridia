from .main import *
from .cross import *
from .dipoles import *
from ..vibrations import *

def main():
    atoms = ase_read(f"{ir_root}/atoms/wollastonite.cif")
    x = iridia(atoms)
    print(f"{x.abs(1) = }")
    x.abs(1)
    return


class iridia:
    def __init__(self, atoms: Atoms, ddm: np.ndarray = None, dyn: np.ndarray = None, method = "central", h: float = 1e-5) -> None:
        self.atoms: Atoms = atoms

        self.method: str = method
        self.h: float = h

        # Dynamical Matrix
        if (dyn is None):
            self.dyn: np.ndarray = dyn
            
        vd = vdosDyn(dyn)
        self.freqk: np.ndarray = vd[0]
        self.vibrations: np.ndarray = vd[1]

        # Dipole Partials
        if (ddm is None):
            ddm = atomDipolePartials(atoms)
    
    def abs(self, w: float, y: float = 0.25):
        print(f"{self.ddm = }")
        return absorbance(w, self.freqk, self.ddm, y)




#main()
