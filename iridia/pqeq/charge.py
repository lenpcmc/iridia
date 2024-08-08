from .main import *
from .energy import *
from .force import *

from scipy.sparse.linalg import spilu, LinearOperator, cg as ConjugateGradient
from scipy.sparse import SparseEfficiencyWarning

from warnings import filterwarnings
filterwarnings("ignore", category = RuntimeWarning)
filterwarnings("ignore", category = SparseEfficiencyWarning)

def main():
    atoms = ase_read("../resources/other/cyclohexane.sdf")

    #q = pqeq(atoms)
    q = 0.
    spos = atoms.positions
    #print(q0)
    for i in range(1):
        pos = atoms.positions
        spos = shellPositions(pos, spos, atoms.get_chemical_symbols(), q, atoms.cell, atoms.pbc)

        #plt.scatter(*pos[...,0:2].T)
        #plt.scatter(*spos[...,0:2].T)
        #plt.show()
        
        q = PQEq(pos, spos, atoms.get_chemical_symbols(), atoms.cell, atoms.pbc, 1)
        atoms.get_charges = lambda: q
        plt.plot(q)
        print(q)
        print(np.sum(q))
    plt.show()
    return


def pqeq(atoms: Atoms, n: int = 0) -> np.ndarray[float]:
    #
    pos: np.ndarray = atoms.positions
    spos: np.ndarray = relaxShells(atoms, n)
    #spos: np.ndarray = atoms.positions

    elem: list[str] = atoms.get_chemical_symbols()

    cell: Cell = atoms.cell
    pbc: list[bool] = atoms.pbc

    return PQEq(pos, spos, elem, cell, pbc, n)


def PQEq(positions: np.ndarray[float], spositions: np.ndarray[float], elem: list[str], cell: Cell = None, pbc: list[bool] = None, n: int = 0) -> np.ndarray[float]:
    """ Charge Equilibrium Condition (Backend) """

    # Shorthand
    pos: np.ndarray = np.array(positions)
    spos: np.ndarray = np.array(spositions)

    # PQEq Parameters
    params = loadParams(n)
    Xo: np.ndarray = np.array([ params["Xo", e] for e in elem ])
    Jo: np.ndarray = np.array([ params["Jo", e] for e in elem ])
    Ks: np.ndarray = np.array([ params["Ks", e] for e in elem ])
    Z: np.ndarray = np.array([ params["Z", e] for e in elem ])
    
    # Legrange Algorithm
    ricjc: np.ndarray = get_distances(pos, pos, cell, pbc)[1]
    ricjs: np.ndarray = get_distances(pos, spos, cell, pbc)[1]

    Cicjc: np.ndarray = C(ricjc, elem, n = n)
    Cicjs: np.ndarray = C(ricjs, elem, n = n)

    #Hij: np.ndarray = 14.4 * Cicjc + np.diag(Jo)
    #Ai: np.ndarray = Xo + Z * np.sum(np.tril( Cicjc - Cicjs ), axis = 1, keepdims = True)

    Hij: np.ndarray = 14.4 * Cicjc + np.diag(Jo)
    Ai: np.ndarray = Xo + 14.4 * Z * np.sum(np.tril( Cicjc - Cicjs ).T, axis = 0, keepdims = True)
    Hinv = np.linalg.inv(Hij)

    # H Inversion (PCG  Approximation)
    sHij_iLU: SuperLu = spilu(Hij)
    M: LinearOperator = LinearOperator( Hij.shape, sHij_iLU.solve )

    # Charge Equilibrium Condition
    #qt: np.ndarray = ConjugateGradient(Hij, -Ai.T, M = M)[0]
    #qh: np.ndarray = ConjugateGradient(Hij, -1. * np.ones(Ai.shape).T, M = M)[0]
    #mu: float = np.sum(qt) / np.sum(qh)
    
    qt: np.ndarray = Hinv @ -Ai.T
    qh = Hinv @ (-1. * np.ones(Ai.shape).T)
    mu = np.sum(qt) / np.sum(qh)

    q: np.ndarray = qt - mu * qh
    return q


if __name__ == "__main__":
    main()
