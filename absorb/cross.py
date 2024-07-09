from .main import *

pi: float = np.pi
ep0: float = 8.854e-12
c: float = 299_792_458

def absorbance(w: float, k: float, ddm: float, y: float = 0.25) -> float:
    # Init
    w: np.ndarray = np.atleast_2d(w)
    k: np.ndarray = np.atleast_2d(k).T
    ddm: np.ndarray = np.atleast_2d(ddm).T
    y: np.ndarray = np.atleast_2d(y).T

    ddm2: np.ndarray = np.sum(ddm**2, axis = 0)

    # Calc
    absCoeff: float = pi / ( 3. * ep0 * c )
    broadening: np.ndarray = y / ( (k - w)**2 + y**2 )

    return np.sum( absCoeff * ddm2 * broadening, axis = 1 )


from ..pqeq.charge import pqeq
from ..vibrations.vdos import vdosDyn

atoms = ase_read("pythonIR/resources/lammps/saved/LiB-30.data", format = "lammps-data")

dyn = np.load("pythonIR/LiB-30.npy")
freqk, vibrations = vdosDyn(dyn)

q = pqeq(atoms)
atoms.get_charges = lambda: q

ddm = dipoleSpectrumAtoms(atoms, vibrations)
ds = absorbance( np.linspace(4000, 100, 2500), freqk, ddm )

print(f"{ds.shape = }")

