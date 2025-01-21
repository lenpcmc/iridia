from .main import *

from .absorb.cross import *
from .absorb.dipoles import *
from .absorb.dspectrum import *
from .absorb.vspect import *

from .vibrations.hess import *
from .vibrations.vdos import *
from .vibrations.vparts import *
from .vibrations.vplot import *
from .vibrations.autohess import *

from .pqeq.pqeq import *
from .pqeq.charge import *
from .pqeq.qeq import qeq

from .visualize.vplot import *
from .visualize.vspect import *

from math import ceil
from pymatgen.io.ase import AseAtomsAdaptor

from collections.abc import Callable
from functools import cache

class iridia:

    def __init__(self, atoms = None, **kwargs) -> None:
        
        self.calc = CHGNetCalculator()
        if isinstance(atoms, Atoms):
            self.atoms = atoms
            self.calc = atoms.calc
        elif isinstance(atoms, str):
            self.atoms = self.read(atoms, **kwargs)
        
        for attr, val in kwargs.items():
            self.__setattr__(attr, val)
        
        self.verbose = None
        
        return


    def read(
            self,
            filename: str,
            format = None,
            relax = True,
            numAtoms: int = 1000,
            repeat: int = None,
            **kwargs,
        ) -> None:
        # Read
        self.atoms = ase_read(filename, format = format)
        self.relax(**kwargs) if relax else self.atoms

        # Build
        if repeat:
            self.atoms = self.atoms * repeat
            self.atoms.calc = self.calc

        elif numAtoms:
            rnum: int = ceil(np.cbrt( numAtoms / len(self.atoms) ))
            self.atoms = self.atoms * rnum
            self.atoms.calc = self.calc

        return self.relax(**kwargs) if relax else self.atoms


    def write(self, filename: str, **kwargs) -> None:
        ase_write(filename, self.atoms, **kwargs)
        return


    def relax(self, opt: str = "LBFGS", **kwargs) -> Atoms:
        
        rlx = relax(self.atoms, opt, **kwargs)

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

        elif attr == "structure":
            result: Structure = self.get_structure()

        else:
            pass

        return result


    def _get_structure(self, atoms: Atoms = None) -> Structure:
        return AseAtomsAdaptor( atoms if atoms is not None else self.atoms )


    def get_dynamical(self, **kwargs) -> np.ndarray:
        if (isinstance(self.atoms.calc, CHGNetCalculator) or self.atoms.calc is None):
            self.dyn = dynamical(self.atoms, self.verbose, autohessian, **kwargs)
        else:
            self.dyn = dynamical(self.atoms, self.verbose, hessian, **kwargs)
        return self.dyn.copy()


    @ensure("dyn")
    def get_vdos(self, **kwargs) -> (np.ndarray, np.ndarray):
        vdos: np.ndarray = vdosDyn(self.dyn)
        self.freqk = vdos[0]
        self.vibrations = vdos[1]
        return vdos


    def get_charges(self, atoms = None, **kwargs) -> np.ndarray:
        
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
    def get_cparts(
            self,
            choose: Callable[[Atoms], list] = lambda atoms: np.zeros(len(atoms)),
            **kwargs,
        ) -> np.ndarray:
        self.cparts = categoryParts(self.atoms, self.vibrations, choose)
        return self.cparts


    @ensure("freqk", "vibrations", "ddm")
    def absorbance(
            self,
            w: float = np.linspace(60, 5, 2000),
            y: float = 0.25,
            choose: Callable[[Atoms], list] = lambda atoms: np.zeros(len(atoms)),
        ) -> float:
        cparts: np.ndarray = self.get_cparts(choose)
        return absorbance(w, self.freqk, self.ddm * np.sqrt(cparts), y)


    @cache
    @ensure("freqk", "vibrations", "ddm")
    def bands(
            self,
            w: float = np.linspace(60, 5, 2000),
            y: float = 0.25,
            choose: Callable[[Atoms], list] = lambda atoms: np.zeros(len(atoms)),
        ) -> np.ndarray[float]:
        # Init
        cats: list = choose(self.atoms)
        cparts: np.ndarray = self.get_cparts(choose)
        cabs: np.ndarray = self.absorbance(w, self.freqk, self.ddm * np.sqrt(cparts), y)

        # Area
        cculm: np.ndarray = np.sum(cabs, axis = 0) * (np.max(w) - np.min(w)) / w.size



    @ensure("freqk", "vibrations", "ddm")
    def plot(
            self,
            w: float = np.linspace(60, 15, 2000),
            y: float = 0.25,
            choose: Callable[[Atoms], list] = lambda atoms: np.zeros(len(atoms)),
            **kwargs,
        ) -> None:

        spect: np.ndarray = self.absorbance(w, y, choose)
        w = extend(w, 2)
        vspect(w, spect.T, **kwargs)
        return


