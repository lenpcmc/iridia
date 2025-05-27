from .main import *

from chgnet.model import CHGNet, CHGNetCalculator
from chgnet.model.model import BatchedGraph
from chgnet.graph import CrystalGraph, CrystalGraphConverter
from pymatgen.io.ase import AseAtomsAdaptor

from torch import Tensor, device
from torch.autograd import grad

from typing import Union

def _loneBatch(graph, chgnet) -> BatchedGraph:
    return BatchedGraph.from_graphs([graph], chgnet.bond_basis_expansion, chgnet.angle_basis_expansion)

def autohessian(atoms: Atoms | Structure | CrystalGraph, model = None, verbose: str = "Computing Hessian Matrix") -> np.ndarray:
    # Ensure
    if not isinstance(atoms.calc, CHGNetCalculator):
        atoms = atoms.copy()
        atoms.calc = CHGNetCalculator()

    # Init
    chgnet: CHGNet = atoms.calc.model if model is None else model
    device: device = next(chgnet.parameters()).device
    N: int = len(atoms)

    #assert isinstance(atoms, Atoms | Structure | CrystalGraph)
    assert isinstance(atoms, Atoms) or isinstance(atoms, Structure) or isinstance(atoms, CrystalGraph)
    if isinstance(atoms, Atoms):
        structure: Structure = AseAtomsAdaptor().get_structure(atoms)
        graph: CrystalGraph = chgnet.graph_converter(structure).to(device)

    elif isinstance(atoms, Structure):
        graph: CrystalGraph = chgnet.graph_converter(atoms).to(device)

    elif isinstance(atoms, CrystalGraph):
        graph: CrystalGraph = atoms.to(device)

    batch: BatchedGraph = _loneBatch(graph, chgnet)
    E: Tensor = chgnet._compute(batch).get('e')
    dE: Tensor = grad( E.sum(), batch.atom_positions, create_graph = True, retain_graph = True)[0].flatten()

    H: list[Tensor] = [ grad( ddE, batch.atom_positions, retain_graph = True )[0] for ddE in tqdm(dE, verbose) ]
    hessian: np.ndarray = N * np.array([ h.flatten().detach().cpu().numpy() for h in H ])

    return hessian

