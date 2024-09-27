from .main import *
from .charge import *
from .energy import *
from .force import *

from ase.calculators.calculator import Calculator, all_changes, all_properties

def main():
    atoms = ase_read("../resources/relaxed/Li2B4O7.cif")
    #atoms = ase_read("../resources/cifs/alphaCristobalite.cif")
    atoms = ase_read("../resources/relaxed/alphaCristobalite.cif")
    atoms = ase_read("../resources/relaxed/wollastonite.cif")
    q = qeq(atoms, 0)
    atoms.get_charges = lambda: q
    #print(f"{q = }")
    return


#class QEqCalculator(Calculator):
class QEqCalculator:
    """ PQEq Calculator for ASE """
    implemented_properties = ( "energy", "forces", "dipole", "charges" )

    def __init__(self, atoms: Atoms, n: int = 0, **kwargs) -> None:
        
        #super().__init__(command, **kwargs)
        self.atoms = atoms
        self.elem: list[str] = atoms.get_chemical_symbols()
        self._par: dict[str,str: float] = loadParams(n)
        self._get_params()

        self.rnorm = atoms.get_all_distances(mic = True)
        self.charges = self.qeq()
        
        return
    
    def calculate(self, atoms: Atoms, properties: list = implemented_properties):
        
        self.pos = atoms.positions()
        self.cell = atoms.cell.copy()
        self.pbc = atoms.pbc.copy()

        if ("energy" in properties):
            PQEqEnergy(pos, spos, symbols, charges, cell, pbc, n)

            #pqeqEnergy(atoms)
            print("energy")
        if ("forces" in properties):
            #pqeqForces(atoms)
            print("forces")
        if ("dipole" in properties):
            #pqeqDipole(atoms)
            print("dipole")
        if ("charges" in properties):
            #pqeq(atoms)
            print("charges")


    def qeq(self) -> np.ndarray:
        pos = self.atoms.positions
        Cicjc = C(self.rnorm, self.elem)
        
        H = 14.4 * Cicjc + np.diag(self.Jo)
        Hinv = np.linalg.inv(H)
        Ai = self.Xo

        qt = Hinv @ -Ai
        qh = Hinv @ (1. * np.ones(Ai.shape))
        mu = np.sum(qt) / np.sum(qh)

        q = qt - mu * qh
        return q


    def _get_params(self, n: int = None) -> list[np.ndarray]:
        # Init
        par: dict[str,str: float] = loadParams(n) if n is not None else self._par

        # Param Arrays
        self.Xo = np.array([ par["Xo", e] for e in self.elem ])
        self.Jo = np.array([ par["Jo", e] for e in self.elem ])

        return [ self.Xo, self.Jo ]


    def qeqEnergy(self, charges: np.ndarray = None) -> np.ndarray:
        charges: np.ndarray = charges if charges is not None else self.charges
        H: np.ndarray = 14.4 * charges @ charges.T / self.rnorm
        return H


def qeq(atoms, n = 0):
    pos = atoms.positions
    elem = atoms.get_chemical_symbols()
    cell = atoms.cell
    pbc = atoms.pbc
    return QEq(pos, elem, cell, pbc, n)


def QEq(pos, elem, cell, pbc, n = 0):

    par = loadParams(n)
    Xo = np.array([ par["Xo", e] for e in elem ])
    Jo = np.array([ par["Jo", e] for e in elem ])

    rnorm = get_distances(pos, pos, cell = cell, pbc = pbc)[1]
    Cicjc = C(rnorm, elem)
    #Cicjc = Cicjc * (rnorm <= 12.5)
    
    Hij = 14.4 * Cicjc + np.diag(Jo)
    Ai = np.array([ Xo ]).T

    # H Inverse (PCG  Approximation)
    sHij_iLU: SuperLu = spilu(Hij)
    M: LinearOperator = LinearOperator( Hij.shape, sHij_iLU.solve )

    # Charge Equilibrium Condition
    qt: np.ndarray = ConjugateGradient(Hij, -Ai, M = M)[0]
    qh: np.ndarray = ConjugateGradient(Hij, -1. * np.ones(Ai.shape), M = M)[0]
    mu: float = np.sum(qt) / np.sum(qh)

    q = np.array([ qt - mu * qh ]).T
    pq = PQEq(pos.copy(), pos.copy(), elem, cell, pbc, n)
    
    print(f"{q = }")
    print(f"{pq = }")
    print(f"{q - pq = }")
    
    plt.plot(q)
    plt.plot(pq)
    plt.show()
    
    return q


#main()
