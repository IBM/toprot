from .atom import atom_lift, node_lift
from .bond import bond_lift, edge_lift
from .molecule import supercell_lift
from .residue import residue_lift
from .ss import ss_lift

LIFTER_REGISTRY = {
    "atom": atom_lift,
    "bond": bond_lift,
    "residue": residue_lift,
    "secondary_structure": ss_lift,
    "supercell": supercell_lift,
}
