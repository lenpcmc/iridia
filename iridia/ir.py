from .main import *

from .absorb.cross import *
from .absorb.dipoles import *
from .absorb.dspectrum import *

from .vibrations import *

from .pqeq.pqeq import *
from .pqeq.charge import *

from .visualize import *

from collections.abc import Callable
from typing import Any

from pymatgen.io.ase import AseAtomsAdaptor


def main():
    x = iridia()
    x.read(f"{ir_root}/resources/atoms/wollastonite.cif")
    x.plot(invert = False)

    w = np.linspace(2000, 400, 1000) * 0.03
    plt.plot(w / 0.03, x.abs(w))
    plt.show()
    return


class iridia:

    def __init__(self, atoms = None, **kwargs) -> None:
        
        atype = type(atoms)
        if issubclass(atype, Atoms):
            self.atoms = atoms
        elif atype == str:
            self.atoms = self.read(atoms, **kwargs)
        
        for attr, val in kwargs.items():
            self.__setattr__(attr, val)
        
        self.verbose = None
        
        return


    def read(self, filename: str, format = None, relax = True, **kwargs) -> None:
        self.atoms = ase_read(filename, format = format)
        self.relax(**kwargs) if relax else 0.
        return self.atoms


    def write(self, filename: str, **kwargs) -> None:
        ase_write(filename, self.atoms, **kwargs)
        return


    def relax(self, opt: str = "LBFGS", fmax: float = 0.01, steps: int = 500, **kwargs) -> Atoms:
        
        rlx = relax(self.atoms, opt, fmax, steps, **kwargs)

        self.structure = rlx[0]
        self.atoms = rlx[1]

        return self.atoms


    def ensure(*reqs):
        def req_decorator(func):
            def wrapper(self, *args, **kwargs):

                # Ensure Reqs Exist
                for r in reqs:
                    if (self.__dict__.get(r) is None):
                        self.__setattr__(r, self._get(r))

                # Result
                x = func(self, *args, **kwargs)

                return x
            return wrapper
        return req_decorator


    def _get(self, attr):
        
        if attr == "dyn":
            result: np.ndarray = self.get_dynamical()
        
        elif attr == "freqk":
            result: np.ndarray = self.get_vdos()[0]

        elif attr == "vibrations":
            result: np.ndarray = self.get_vdos()[1]

        elif attr == "charges":
            result: np.ndarray = self.get_charges()

        elif attr == "ddm":
            result: np.ndarray = self.get_ddm()

        elif attr == "struct":
            result: Structure = self.get_struct()

        else:
            pass

        return result


    def _get_struct(self, atoms: Atoms = None) -> Structure:
        return AseAtomsAdaptor( atoms if bool(atoms) else self.atoms )


    def get_dynamical(self, **kwargs) -> np.ndarray:
        self.dyn = dynamical(self.atoms, self.verbose, **kwargs)
        return self.dyn.copy()


    @ensure("dyn")
    def get_vdos(self, **kwargs) -> (np.ndarray, np.ndarray):
        vd: np.ndarray = vdosDyn(self.dyn)
        self.freqk = vd[0]
        self.vibrations = vd[1]
        return vd


    def get_charges(self, **kwargs) -> np.ndarray:
        
        if ("charges" not in self.atoms.calc.implemented_properties):
            self.charges = pqeq(self.atoms)
            self.atoms.get_charges = lambda: self.charges

        else:
            self.charges = self.atoms.get_charges()

        return self.charges
    

    @ensure("vibrations", "charges")
    def get_ddm(self, **kwargs) -> np.ndarray:
        ddm = dipoleSpectrumAtoms( self.atoms, self.vibrations, self.verbose, **kwargs )
        self.ddm = extend(ddm, 3)
        return self.ddm
    

    @ensure("vibrations")
    def get_cparts(self, choose: Callable[Atom] = lambda a: 0., **kwargs) -> np.ndarray:
        return categoryParts(self.atoms, self.vibrations, choose)


    @ensure("freqk", "vibrations", "ddm")
    def absorbance(self, w: float, y: float = 0.25, **kwargs) -> float:
        cparts: np.ndarray = self.get_cparts(**kwargs)
        return absorbance(w, self.freqk, self.ddm * np.sqrt(cparts), y)


    @ensure("freqk", "vibrations", "ddm")
    def plot(self, w = np.linspace(2000, 0, 2000) * 0.03, **kwargs) -> None:
        #fig,ax = plt.subplots()
        #spectrumPlot(ax, w, self.abs(w, **kwargs), **kwargs)
        #plt.show()
        vspect(w, self.abs(w, **kwargs), **kwargs)
        return


    



#main()
