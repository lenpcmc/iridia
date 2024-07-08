from .main import *


def main():
    structure, atoms = buildArray(f"{aroot}/wollastonite.cif", 1, fmax = 1)
    structure, atoms = buildArray(f"{aroot}/danburite.cif", 3, fmax = 1)
    structure, atoms = buildArray(f"{aroot}/betaCristobalite.cif", 1, fmax = 1)
    freqk, vibrations = vdos(atoms)
    return


def vdos(atoms: Atoms, conv: float = 15.63, h: float = 1e-5) -> (np.ndarray, np.ndarray):
    """ Get the frequencies and vibrations of phonons """
    """ for a given set of atoms. """
    
    # VDOS Calculation
    dyn: np.ndarray = dynamical(atoms, h)
    freqk, vibrations = np.linalg.eigh(dyn)
    freqk *= conv

    # Remove nan
    np.nan_to_num(freqk, 0.)
    np.nan_to_num(vibrations, 0.)

    return freqk, vibrations


def dynamical(atoms: Atoms, h: float = 1e-5):
    """ Get the Dynamical (Hessian * sqrt(m1*m2)) """
    """ for a given set of atoms. """

    mtensor: np.ndarray = 1. / np.array([ (m, m, m) for m in atoms.get_masses() ]).reshape((atoms.positions.size, 1))
    mmask: np.ndarray = np.sqrt(mtensor @ mtensor.T)
    H: np.ndarray = hessian(atoms, h)
    return H * mmask


def hessian(atoms: Atoms, h: float = 1e-5):
    """ Get the Hessian (dE/dn,dm) for """
    """ a given set of atoms. """

    # Ensure Calculator
    if (atoms.calc is None):
        atoms.calc = CHGNetCalculator()

    # Allocate and Fill
    H: np.ndarray = np.array([ hessRow(atoms, i, h) for i in trange( 3*len(atoms) ) ])

    # Ensure Symmetry over Numeric Precision
    return -1. * (H + H.T) / 2.


def hessRow(atoms: Atoms, i: int, h: float = 1e-5) -> np.ndarray:
    """ Get the ith row of the Hessian """
    """ for a given set of atoms """

    # Shorthand
    apos: np.ndarray = atoms.positions
    
    # Finite Forward Difference
    apos[i // 3, i % 3] += h
    fp: np.ndarray = atoms.get_forces().flatten()

    # Finite Backward Difference
    apos[i // 3, i % 3] -= h * 2.
    fn: np.ndarray = atoms.get_forces().flatten()

    # Central Difference Derivative
    apos[i // 3, i % 3] += h
    D: np.ndarray = (fp - fn) / (2. * h)

    return D


if __name__ == "__main__":
    main()
