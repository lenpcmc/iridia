from .main import *

pi: float = np.pi
ep0: float = 8.854e-12
c: float = 299_792_458

def absorbance(w: float, k: float, ddm: float, y: float = 0.25) -> float:
    # Init
    w: np.ndarray = extend(w, 3).swapaxes(0,1)
    k: np.ndarray = extend(k, 3)
    ddm: np.ndarray = extend(ddm, 3)
    y: np.ndarray = extend(y, 3)

    ddm2: np.ndarray = np.sum(ddm**2, axis = 1, keepdims = True)

    # Calc
    absCoeff: float = pi / ( 3. * ep0 * c )
    broadening: np.ndarray = y / ((k - w)**2 + y**2)

    return np.sum( absCoeff * ddm2 * broadening, axis = 0 )


'''
from ..pqeq.charge import pqeq
from ..vibrations.vdos import dynamical, vdosDyn
from .dSpectrum import *
from ..visualize.vspect import *
from vdos import vdos
from psvector import crossIR, dipolePartials, dipole as dp
from rruffIR import *

spectrum, rs = rruffIR("resources/rruff/processed/data/Wollastonite__R040008-1__Infrared__Infrared_Data_Processed__1001.txt")
spectrum *= 0.03

#structure, atoms = buildArray("resources/cifs/wollastonite.cif", 2, 0.001)
atoms = ase_read("resources/cifs/wollastonite.cif")

relaxer = StructOptimizer(optimizer_class = "LBFGS")
x = relaxer.relax(atoms).get("trajectory").atoms

q = pqeq(atoms)
atoms.get_charges = lambda: q

dyn = dynamical(atoms.copy())
#dyn = np.load("resources/arrays/wollastonite.npy")
freqk, vibrations = vdosDyn(dyn)

ddm = dipoleSpectrumAtoms(atoms, vibrations)
ds = absorbance(spectrum, freqk, ddm)

fig,ax = plt.subplots()
spectrumPlot(ax, spectrum, ds, label = "iridia")
spectrumPlot(ax, spectrum, rs, label = "exp", invert = True)
plt.legend()
plt.show()
'''
