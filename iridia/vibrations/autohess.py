from .main import *

import torch
from chgnet.model import CHGNet, CHGNetCalculator
from chgnet.graph import CrystalGraph, CrystalGraphConverter
from pymatgen.io.ase import AseAtomsAdaptor

from typing import Union

def main():
    from ..ir import iridia
    from .hess import dynamical
    from .vdos import vdosDyn
    from ..visualize.vplot import vdosPlot, vdosDist
    struct, atoms = buildArray(f"../resources/cifs/alphaCristobalite.cif", 1)
    
    mtensor: np.ndarray = np.array([ (m, m, m) for m in atoms.get_masses() ]).reshape((atoms.positions.size, 1))
    mmask: np.ndarray = 1. / np.sqrt(mtensor @ mtensor.T)
    
    def hist(a, **kwargs):
        y,x = np.histogram(a, **kwargs)
        return x[:-1], y
    
    H = autohessian(atoms)
    D = H * mmask
    D2 = dynamical(atoms)

    print(f"{D = }")
    print(f"{D2 = }")
    print(f"{D / D2 = }")
    print(f"{D2 / D = }")

    k, vib = vdosDyn(D)
    k2, vib2 = vdosDyn(D2)

    #fig,ax = vdosPlot()
    #ax.plot(*vdosDist(k, width = 50))
    #ax.plot(*vdosDist(k2, width = 50))
    plt.show()
    return


def autohessian(atoms: Union[Atoms, Structure, CrystalGraph], model = None):
    # Init
    chgnet = atoms.calc.model if model is None else model
    device = next(chgnet.parameters()).device
    N = len(atoms)

    #assert isinstance(atoms, Atoms | Structure | CrystalGraph)
    assert isinstance(atoms, Atoms) or isinstance(atoms, Structure) or isinstance(atoms, CrystalGraph)
    if isinstance(atoms, Atoms):
        structure = AseAtomsAdaptor().get_structure(atoms)
        graph = chgnet.graph_converter(structure)

    elif isinstance(atoms, Structure):
        graph = chgnet.graph_converter(atoms)

    elif isinstance(atoms, CrystalGraph):
        graph = atoms

    graph = graph.to(device)
    lattice = graph.lattice
    linv = torch.linalg.inv(lattice)

    lv = lattice.sum(axis = 1, keepdims = True)
    lt = torch.concatenate([ lv for i in range(N) ], axis = 0)
    ltd = 1./lt * torch.eye(3*N).to(device)
    
    print(f"{lt = }")
    print(f"{1/2.*lt = }")
    print(f"{ltd = }")
    print(f"{ltd.T @ torch.eye(3*N).to(device) @ ltd = }")

    def compute_energy(graph_positions):
        graph.atom_frac_coord = graph_positions @ linv
        #graph.atom_frac_coord = graph_positions
        return chgnet.forward([graph], task = 'e').get('e')

    #hessian = torch.autograd.functional.hessian(compute_energy, graph.atom_frac_coord @ lattice)
    hessian = torch.autograd.functional.hessian(compute_energy, graph.atom_frac_coord @ lattice).reshape(3*N,3*N)
    #hessian = torch.autograd.functional.hessian(compute_energy, graph.atom_frac_coord).reshape(3*N,3*N)
    ahessian = ltd.T @ hessian @ ltd
    #return hessian.detach().cpu().numpy()
    return ahessian.detach().cpu().numpy()

