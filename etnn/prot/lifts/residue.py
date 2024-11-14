import torch
from torch_geometric.data import Data
from torch_geometric.utils import one_hot
from typing import List


from etnn.combinatorial_data import Cell

NUM_FEATURES = 24


def residue_lift(graph: Data) -> set[Cell]:
    """
    Identify residues in a graph.

    This function returns the residues of the given graph. Each residue is represented as a tuple
    containing a singleton set of residue index and a feature vector.

    Parameters
    ----------
    graph : Data
        The input graph represented as a PyTorch Geometric Data object.

    Returns
    -------
    set[Cell]
        A set of tuples, where each tuple is a singleton set of residue index and a feature vector.

    Raises
    ------
    ValueError
        If the input graph does not contain a feature matrix 'x'.

    Notes
    -----
    The function converts the input graph to an RDKit molecule and then iterates over each atom
    to create the atom nodes. Atom features are computed using the `compute_atom_features` function.

    Attributes
    ----------
    num_features : int
        The number of features for each atom.
    """
    residue_features = get_residue_features(graph.residues)
    nodes = {(frozenset([node]), tuple(residue_features[node])) for node in range(len(graph.residues))}
    return nodes

residue_lift.num_features = NUM_FEATURES

def get_residue_features(residues: List[str]) -> tuple[float]:
    """
    Computes features for residues.

    Parameters:
        residues (List[str]): List of three-letter residue codes.

    Returns:
        Tensor: One-hot encoded residue types concatenated with physicochemical properties.
    """
    # Define 20 standard amino acids
    standard_aas = [
        'ALA', 'CYS', 'ASP', 'GLU', 'PHE',
        'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
        'MET', 'ASN', 'PRO', 'GLN', 'ARG',
        'SER', 'THR', 'VAL', 'TRP', 'TYR'
    ]

    # Map residue names to indices
    residue_type_dict = {res: i for i, res in enumerate(standard_aas)}
    num_residue_types = len(standard_aas)

    # Handle non-standard residues by assigning them to a special 'UNK' category
    residue_indices = [
        residue_type_dict.get(res, num_residue_types) for res in residues
    ]
    # Add 'UNK' category if necessary
    if num_residue_types not in residue_type_dict.values():
        num_classes = num_residue_types + 1
    else:
        num_classes = num_residue_types

    # One-hot encode residue types
    residue_one_hot = one_hot(torch.tensor(residue_indices), num_classes=num_classes).float()

    # Define physicochemical properties for standard amino acids
    # Properties sourced from the AAindex database (https://www.genome.jp/aaindex/)
    # Here, we define a subset for demonstration
    physicochemical_properties = {
        'ALA': [1.8, 0.5, 0.0],
        'CYS': [2.5, 8.3, -1.0],
        'ASP': [-3.5, 13.0, -2.0],
        'GLU': [-3.5, 12.3, -2.0],
        'PHE': [2.8, 0.0, 0.0],
        'GLY': [-0.4, 0.0, 0.0],
        'HIS': [-3.2, 10.4, -1.0],
        'ILE': [4.5, 0.0, 0.0],
        'LYS': [-3.9, 11.3, -1.0],
        'LEU': [3.8, 0.0, 0.0],
        'MET': [1.9, 0.0, 0.0],
        'ASN': [-3.5, 11.6, -1.0],
        'PRO': [-1.6, 0.0, 0.0],
        'GLN': [-3.5, 10.5, -1.0],
        'ARG': [-4.5, 12.5, -1.0],
        'SER': [-0.8, 0.0, 0.0],
        'THR': [-0.7, 0.0, 0.0],
        'VAL': [4.2, 0.0, 0.0],
        'TRP': [-0.9, 0.0, 0.0],
        'TYR': [-1.3, 0.0, 0.0],
        'UNK': [0.0, 0.0, 0.0]  # For non-standard residues
    }

    # Extract physicochemical properties
    phys_props = []
    for res in residues:
        props = physicochemical_properties.get(res, physicochemical_properties['UNK'])
        phys_props.append(props)
    phys_props = torch.tensor(phys_props, dtype=torch.float)

    # Concatenate one-hot and physicochemical properties
    residue_features = torch.cat([residue_one_hot, phys_props], dim=-1)

    return tuple(residue_features.tolist())
