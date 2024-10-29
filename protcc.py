import os
import os.path as osp
import sys
from typing import Callable, List, Optional, Dict

import torch
from torch import Tensor
from torch_geometric.data import Data, download_url, extract_zip, InMemoryDataset
from torch_geometric.utils import one_hot, scatter
from tqdm import tqdm
import yaml

from etnn.lifter import Lifter, get_adjacency_types
from etnn.qm9.lifts.registry import LIFTER_REGISTRY
from etnn.lifter import CombinatorialComplexTransform

from Bio.PDB import PDBParser, DSSP
import numpy as np

class ProteinCombinatorialComplexDataset(InMemoryDataset):
    """
    Dataset for building a combinatorial complex from protein structures.
    The first layer consists of residues, and the second layer consists of secondary structures.
    """

    def __init__(
        self,
        root: str,
        lifters: List[str],
        neighbor_types: List[str],
        connectivity: str,
        pdb_dir: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        **lifter_kwargs,
    ) -> None:
        self.lifters = lifters
        self.neighbor_types = neighbor_types
        self.connectivity = connectivity
        self.pdb_dir = pdb_dir  # Directory containing PDB files

        # Initialize lifter and adjacencies
        self.dim = 2  # Two layers: residues and secondary structures
        self.adjacencies = get_adjacency_types(
            self.dim,
            connectivity,
            neighbor_types,
        )
        self.lifter = Lifter(self.lifters, LIFTER_REGISTRY, self.dim, **lifter_kwargs)

        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        # Include 'pdb_ids.txt' as a required raw file
        pdb_ids_file = 'pdb_ids.txt'
        pdb_files = [f for f in os.listdir(self.pdb_dir) if f.endswith('.pdb')]
        return [pdb_ids_file] + pdb_files
    

    @property
    def processed_file_names(self) -> str:
        return "protein_data.pt"

    def download(self) -> None:
        """
        Download PDB files based on a list of PDB IDs provided in 'pdb_ids.txt'.
        The PDB files are saved in the 'pdb_dir' directory.
        """
        from Bio.PDB import PDBList
        import gzip
        import shutil

        pdb_ids_file = osp.join(self.raw_dir, 'pdb_ids.txt')
        
        if not osp.exists(pdb_ids_file):
            raise FileNotFoundError(f"PDB IDs file not found: {pdb_ids_file}")
        
        # Read PDB IDs from 'pdb_ids.txt'
        with open(pdb_ids_file, 'r') as f:
            pdb_ids = [line.strip().lower() for line in f if line.strip()]
        
        if not pdb_ids:
            raise ValueError("No PDB IDs found in 'pdb_ids.txt'. Please provide at least one PDB ID.")
        
        pdbl = PDBList()
        
        # Ensure pdb_dir exists
        os.makedirs(self.pdb_dir, exist_ok=True)
        
        for pdb_id in tqdm(pdb_ids, desc="Downloading PDB files"):
            try:
                downloaded_path = pdbl.retrieve_pdb_file(pdb_id, pdir=self.pdb_dir, file_format='pdb')
                if downloaded_path.endswith('.gz'):
                    with gzip.open(downloaded_path, 'rb') as f_in:
                        with open(osp.join(self.pdb_dir, f"{pdb_id}.pdb"), 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(downloaded_path) 
            except Exception as e:
                print(f"Failed to download PDB ID {pdb_id.upper()}: {e}", file=sys.stderr)


    def process(self) -> None:
        parser = PDBParser(QUIET=True)
        data_list = []

        for pdb_file in tqdm(self.raw_file_names, desc="Processing PDB files"):
            pdb_path = osp.join(self.pdb_dir, pdb_file)
            structure = parser.get_structure(pdb_file, pdb_path)

            # Use the first model
            model = structure[0]

            # Assign secondary structure using DSSP
            try:
                dssp = DSSP(model, pdb_path, dssp='mkdssp')
            except Exception as e:
                print(f"DSSP failed for {pdb_file}: {e}", file=sys.stderr)
                continue

            residues = []
            ss_labels = []
            coords = []
            for chain in model:
                for residue in chain:
                    if residue.id[0] != ' ':  # Skip hetero residues
                        continue
                    resname = residue.get_resname()
                    residues.append(resname)
                    ss = dssp.get((chain.id, residue.id))
                    if ss is None:
                        ss_label = 'C'  # Coil as default
                    else:
                        ss_label = ss[2]  # Secondary structure
                    ss_labels.append(ss_label)
                    # Get alpha carbon coordinates
                    if 'CA' in residue:
                        ca = residue['CA'].get_coord()
                        coords.append(ca)
                    else:
                        coords.append([0.0, 0.0, 0.0])  # Placeholder

            if not residues:
                continue  # Skip if no valid residues

            num_residues = len(residues)
            pos = torch.tensor(coords, dtype=torch.float)

            # Encode residue types
            residue_features = self.get_residue_features(residues)
            x1 = residue_features

            # Encode secondary structures
            ss_features = self.get_secondary_structure_features(ss_labels)
            x2 = ss_features

            # Combine features
            x = torch.cat([x1, x2], dim=-1)

            # Create edge_index based on spatial proximity (e.g., within 8 Å)
            # You can customize this as needed
            threshold = 8.0
            distances = torch.cdist(pos, pos, p=2)
            edge_index = (distances < threshold).nonzero(as_tuple=False).t()
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]  # Remove self-loops

            # Optionally, add edge attributes (e.g., distance)
            edge_attr = distances[edge_index[0], edge_index[1]].unsqueeze(-1)
            y = torch.tensor([0.0])  # Dummy target

            data = Data(
                x=x,
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                pdb_id=pdb_file.replace('.pdb', ''),
            )

            # Apply combinatorial complex transformation
            data = CombinatorialComplexTransform(
                lifter=self.lifter,
                adjacencies=self.adjacencies,
            )(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        # Save processed data
        self.save(data_list, self.processed_paths[0])

    def get_residue_features(self, residues: List[str]) -> Tensor:
        """
        Extract features for residues.

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

        return residue_features

    def get_secondary_structure_features(self, ss_labels: List[str]) -> Tensor:
        """
        Extract features for secondary structures.

        Parameters:
            ss_labels (List[str]): List of secondary structure labels.

        Returns:
            Tensor: One-hot encoded secondary structure types concatenated with structural properties.
        """
        # Define secondary structure types
        ss_types = ['H', 'E', 'C']  # Helix, Sheet, Coil
        ss_type_dict = {ss: i for i, ss in enumerate(ss_types)}
        num_ss_types = len(ss_types)

        # Encode secondary structure labels
        ss_indices = [ss_type_dict.get(ss, 2) for ss in ss_labels]  # Default to 'C' if unknown
        ss_one_hot = one_hot(torch.tensor(ss_indices), num_classes=num_ss_types).float()

        # Define structural properties for secondary structures (for now dummy values)
        # We may want to include properties like hydrogen bonding propensity, flexibility, solvent accessibility
        structural_properties = {
            'H': [1.0, 0.0, 0.5],  # Helix: High hydrogen bonding, low flexibility, moderate solvent accessibility
            'E': [0.8, 0.2, 0.4],  # Sheet: High hydrogen bonding, slightly more flexible
            'C': [0.0, 1.0, 0.9]   # Coil: No hydrogen bonding, high flexibility, high solvent accessibility
        }

        structural_feats = []
        for ss in ss_labels:
            props = structural_properties.get(ss, structural_properties['C'])
            structural_feats.append(props)
        structural_feats = torch.tensor(structural_feats, dtype=torch.float)

        # Concatenate one-hot and structural properties
        ss_features = torch.cat([ss_one_hot, structural_feats], dim=-1)

        return ss_features
    


from torch_geometric.loader import DataLoader

# Define parameters
root = './data/proteins'
pdb_dir = './data/pdb_files'  
lifters = ['residue', 'secondary_structure']
neighbor_types = ['spatial']  
connectivity = 'full' 

# Initialize dataset
dataset = ProteinCombinatorialComplexDataset(
    root=root,
    lifters=lifters,
    neighbor_types=neighbor_types,
    connectivity=connectivity,
    pdb_dir=pdb_dir,
    force_reload=False
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
    print(batch)
   