from .main import *
from ase import Atom
from collections.abc import Callable

def main():
    from .vdos import vdos, vdosDyn
    from .vplot import vplot
    from ..visualize.vspect import vspect, spectrumPlot
    from ..absorb.dipoles import dipole, dipolePartial, atomDipolePartial
    from ..absorb.dSpectrum import dipoleSpectrumAtoms
    from ..absorb.cross import absorbance
    w = np.linspace(1500, 0, 4500) * 0.03
    structure, atoms = buildArray(f"{aroot}/wollastonite.cif", 1, fmax = 1.)
    #structure, atoms = buildArray(f"{aroot}/danburite.cif", 3, fmax = 1)
    #structure, atoms = buildArray(f"{aroot}/betaCristobalite.cif", 1, fmax = 1)
    #structure, atoms = buildArray(f"{aroot}/betaCristobalite.cif", [3,2,2], fmax = 1)
    atoms.get_charges = lambda: atoms.arrays.get("oxi_states")
    
    #freqk, vibrations = vdos(atoms)
    dyn = np.load("wDyn.npy")
    freqk, vibrations = vdosDyn(dyn)

    ddm = dipoleSpectrumAtoms(atoms, vibrations)
    print(f"{ddm.shape = }")

    cparts = categoryParts(atoms, vibrations)
    print(f"{cparts = }")
    print(f"{cparts.shape = }")

    dmparts = np.linalg.norm(ddm, axis = 1, keepdims = True) * cparts
    #dmparts = ddm * cparts
    print(f"{np.sum(dmparts[...,0]) = }")
    print(f"{np.sum(dmparts[...,1]) = }")
    print(f"{np.sum(dmparts[...,2]) = }")
    print(f"{np.sum(np.linalg.norm(ddm, axis = 1)) = }")
    print(f"{dmparts.shape = }")

    ds = absorbance(w, freqk, dmparts)
    print(f"{ds.shape = }")

    fig,ax = plt.subplots()
    spectrumPlot(ax, w, ds, np.unique(atoms.numbers))
    spectrumPlot(ax, w, absorbance(w, freqk, ddm), np.unique(atoms.numbers))
    plt.show()
    #vspect(w, ds, "Si", "O" )

    return


def vparts(vibrations: np.ndarray) -> np.ndarray:
    return np.sum(vibrations**2, axis = -1)


def categorize(atoms: Atoms, choose: Callable[Atom] = lambda a: a.number) -> np.ndarray[bool]:
    assign: list = [ choose(a) for a in atoms ]
    cmask: np.ndarray = np.array([ [ assign[i] == cat for i,a in enumerate(assign) ] for cat in np.unique(assign) ])
    return cmask


def categoryParts(atoms: Atoms, vibrations: np.ndarray, choose: Callable[Atom] = lambda a: a.number) -> np.ndarray:
    parts: np.ndarray = extend(vparts(vibrations), 2)
    cats: np.ndarray = extend(categorize(atoms, choose), 3)

    cparts = list()
    for c in cats:
        print(parts.T.shape)
        print(c.shape)
        print(c)
        cparts.append( c * parts.T )
    cparts = np.array(cparts)
    print(f"{cparts = }")
    print(f"{cparts.shape = }")
    #print(f"{cats = }")
    #print(f"{cats.shape = }")
    #return np.sum(cats * parts, axis = 1)
    #exit()
    return np.sum(cparts, axis = 1).T



main()
