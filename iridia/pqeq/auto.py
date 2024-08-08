from .main import *

import torch
from torch import Tensor, tensor, device, autograd

def main():
    p = PQEqTensor(1)
    print(p.__dict__)
    p._C()
    return


class PQEqTensor:

    def __init__(self, atoms: Atoms, rendition: int = 0, **kwargs) -> None:
        #self.atoms = atoms.copy()
        #self.elem = atoms.get_chemical_symbols()
        
        #self.pos = tensor(atoms.positions)
        #self.cell = atoms.cell.copy()
        #self.pbc = atoms.pbc.copy()
        self.p = True

        return


    def resolve(func):
        def rfunc(self, *args, **kwargs):
            members: list[str] = [ m for m in self.__dir__() if not (m.startswith("__") and m.endswith("__")) ]
            attrs: dict[str] = { m: self.__getattribute__(m) for m in members }
            return func( self, *args, **(attrs | kwargs) )
        return rfunc


    def _get_params(self, n: int = None, **kwargs) -> list[Tensor]:
        # Init
        self._par: dict[str,str: float] = loadParams(n) if n is not None else self._par
        if (n is not None):
            self._par = loadParams(n)
        
        # Electronegativity
        Xo: Tensor = tensor([ [self._par["Xo", e]] for e in self.elem ])
        self.Xo = Xo.to(self.device)

        # Idempotential
        Jo: Tensor = tensor([ [self._par["Jo", e]] for e in self.elem ])
        self.Jo = Jo.to(self.device)

        # Shell Charge
        Z: Tensor = tensor([ [self._par["Z", e]] for e in self.elem ])
        self.Z = Z.to(self.device)

        # Atom sizes
        Rc: Tensor = tensor([ [self._par["Rc", e]] for e in self.elem ])
        Rs: Tensor = tensor([ [self._par["Rs", e]] for e in self.elem ])

        self.Rc = Rc.to(self.device)
        self.Rs = Rs.to(self.device)

        # Spring constants
        Ks: Tensor = tensor([ [self._par["Ks", e]] for e in self.elem ])
        self.Ks = Ks.to(self.device)

        return [ Xo, Jo, Z, Rc, Rs, Ks ]


    def _get_alpha(self, lambda_pqeq: float = 0.462770, **kwargs) -> Tensor:
        alpha: Tensor = 0.5 * lambda_pqeq / self.Rc**2.
        self.alpha = alpha
        return alpha


    @resolve
    def _C(self, ai: Atoms = None, aj: Atoms = None, **kwargs) -> Tensor:
        print(kwargs)
        print(self._C)
        print(self._C.__kwdefaults__)
        _get_alpha
        # Coeffs
        alpha: Tensor = (self.alpha @ self.alpha.T) / (self.alpha + self.alpha.T)
        ralpha: Tensor = torch.sqrt(alpha)
        
        # Interaction Energies
        #eGaussCharges: Tensor = torch.erf(ralpha * self.rnorm) / rnorm
        return 0.
        



    


if __name__ == "__main__":
    main()
