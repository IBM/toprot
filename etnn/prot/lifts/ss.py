import torch
from torch_geometric.data import Data
from torch_geometric.utils import one_hot
from typing import List


from etnn.combinatorial_data import Cell

NUM_FEATURES = 6


def ss_lift(graph: Data) -> set[Cell]:
    """
    Identify secondary structure elements in a graph.

    This function returns the ss lifts of the given graph as a set of cells. Each cell
    represents a ss element and consists of a frozenset of node indices and a feature vector

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

    ss_elements = graph.ss_elements
    ss_features = get_secondary_structure_features(ss_elements)

    cells = set()
    for ind, ss_type in enumerate(ss_elements):
        for el in ss_elements[ss_type]:
            cells.add((frozenset(el), tuple(ss_features[ind])))

    return cells

ss_lift.num_features = NUM_FEATURES

def get_secondary_structure_features(ss_elements: dict) -> tuple[float]:
        """
        Extract features for secondary structures.

        Parameters:
            ss_elements (dict): Dictionary of secondary structure elements.

        Returns:
            Tensor: One-hot encoded secondary structure types concatenated with structural properties.
        """

        #one-hot encoding of ss types (keys of ss_elements)
        ss_types = list(ss_elements.keys())
        ss_type_dict = {ss: i for i, ss in enumerate(ss_types)}

        num_ss_types = len(ss_types)
        ss_indices = [ss_type_dict.get(ss, num_ss_types) for ss in ss_elements]  # Default to 'C' if unknown
        ss_one_hot = one_hot(torch.tensor(ss_indices), num_classes=num_ss_types).float()

        # Define structural properties for secondary structures (for now dummy values)
        # We may want to include properties like hydrogen bonding propensity, flexibility, solvent accessibility
        structural_properties = {
            'H': [1.0, 0.0, 0.5],  # Helix: High hydrogen bonding, low flexibility, moderate solvent accessibility
            'E': [0.8, 0.2, 0.4],  # Sheet: High hydrogen bonding, slightly more flexible
            'C': [0.0, 1.0, 0.9]   # Coil: No hydrogen bonding, high flexibility, high solvent accessibility
        }

        structural_feats = []
        for ss in ss_types:
            props = structural_properties.get(ss, structural_properties['C'])
            structural_feats.append(props)
        structural_feats = torch.tensor(structural_feats, dtype=torch.float)

        # Concatenate one-hot and structural properties
        ss_features = torch.cat([ss_one_hot, structural_feats], dim=-1)

        return tuple(ss_features.tolist())
    
   






