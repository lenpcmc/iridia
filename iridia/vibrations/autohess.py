from .main import *

import torch
from chgnet.model import CHGNet, CHGNetCalculator
from chgnet.graph import CrystalGraph, CrystalGraphConverter
from pymatgen.io.ase import AseAtomsAdaptor

from typing import Union

def main():
    chgnet = CHGNet()
    atoms = ase_read(f"Downloads/alphaQuartz.cif")
    atoms = atoms * 3
    atoms.calc = CHGNetCalculator(chgnet)
    
    mtensor: np.ndarray = np.array([ (m, m, m) for m in atoms.get_masses() ]).reshape((atoms.positions.size, 1))
    mmask: np.ndarray = 1. / np.sqrt(mtensor @ mtensor.T)
    
    def hist(a, **kwargs):
        y,x = np.histogram(a, **kwargs)
        return x[:-1], y
    
    H = auto_hessian(atoms)
    D = H * mmask
    D2 = dynamical(atoms)

    print(f"{D = }")
    print(f"{D2 = }")
    print(f"{D / D2 = }")

    k, vib = vdosDyn(D)
    k2, vib2 = vdosDyn(D2)

    plt.plot(*hist(k, bins = 50))
    plt.plot(*hist(k2, bins = 50))
    plt.show()
    return


def autohessian(atoms: Union[Atoms, Structure, CrystalGraph], model = None):
    # Init
    chgnet = atoms.calc.model if model is None else model

    #assert isinstance(atoms, Atoms | Structure | CrystalGraph)
    assert isinstance(atoms, Atoms) or isinstance(atoms, Structure) or isinstance(atoms, CrystalGraph)
    if isinstance(atoms, Atoms):
        adaptor = AseAtomsAdaptor()
        converter = CrystalGraphConverter()

        structure = adaptor.get_structure(atoms)
        graph = converter(structure)
        
        N = len(atoms)

    elif isinstance(atoms, Structure):
        converter = CrystalGraphConverter()
        graph = converter(atoms)
        
        N = len(atoms)

    elif isinstance(atoms, CrystalGraph):
        graph = atoms
        N = len(graph.atomic_number)

    def compute_energy(graph_positions):
        graph.atom_frac_coord = graph_positions
        return chgnet.forward([graph], task = 'e').get('e')

    hessian = torch.autograd.functional.hessian(compute_energy, graph.atom_frac_coord @ graph.lattice)
    hessian = torch.autograd.grad(force, retain_graph = True)
    return hessian.detach().numpy().reshape( 3*N, 3*N )

