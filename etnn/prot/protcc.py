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

from Bio.PDB import PDBParser, DSSP
import numpy as np

from etnn.lifter import Lifter, get_adjacency_types
from etnn.prot.lifts.registry import LIFTER_REGISTRY
from etnn.lifter import CombinatorialComplexTransform


class ProtCC(InMemoryDataset):
    """
    Dataset for building a CC from protein structures.
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
        self.dim = len(self.lifters) - 1  # Two layers: residues and secondary structures
        self.adjacencies = get_adjacency_types(
            self.dim,
            connectivity,
            neighbor_types,
        )
        self.lifter = Lifter(self.lifters, LIFTER_REGISTRY, self.dim, **lifter_kwargs)

        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        # Include 'pdb_ids.txt' as a required raw file
        # pdb_ids_file = 'pdb_ids.txt'
        pdb_files = [f for f in os.listdir(self.pdb_dir) if f.endswith('.pdb')]
        return pdb_files
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
                print(f"Downloading PDB ID {pdb_id.upper()}...")
                downloaded_path = pdbl.retrieve_pdb_file(pdb_id, pdir=self.pdb_dir, file_format='pdb')
                print(downloaded_path)
                # Rename .ent files to .pdb
                if downloaded_path.endswith('.ent'):
                    new_path = downloaded_path.replace('.ent', '.pdb')
                    os.rename(downloaded_path, new_path)
                    downloaded_path = new_path
                
                if downloaded_path.endswith('.gz'):
                    with gzip.open(downloaded_path, 'rb') as f_in:
                        with open(osp.join(self.pdb_dir, f"{pdb_id}.pdb"), 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(downloaded_path) 
            except Exception as e:
                print(f"Failed to download PDB ID {pdb_id.upper()}: {e}", file=sys.stderr)
    
    
    def group_ss_elements(self, dssp_data, target_ss_list=['H', 'E', 'C']):
        """
        Groups residues into secondary structure (SS) elements based on each target SS code,
        keeping only the residue indices and filtering to keep only the largest supersets.

        Parameters:
            dssp_data (list): List of DSSP tuples.
            target_ss_list (list): List of SS codes to group (e.g., ['H', 'E', 'C']).

        Returns:
            dict: Dictionary where keys are SS codes and values are lists of SS elements,
                each as a list of residue indices, with overlapping subsets removed.
        """
        ss_elements_dict = {ss_code: [] for ss_code in target_ss_list}

        for target_ss in target_ss_list:
            current_element = []
            current_chain = None
            last_residue_number = None
            for entry in dssp_data:
                chain_id, residue_id, aa, ss, *rest = entry
                residue_index = residue_id[1]  # Extract residue index
                if ss == target_ss:
                    if not current_element:
                        # Start of a new SS element
                        current_element = [residue_index]
                        current_chain = chain_id
                        last_residue_number = residue_index
                    else:
                        # Check if current residue is contiguous with the last residue in current_element
                        if (chain_id == current_chain) and (residue_index == last_residue_number + 1):
                            current_element.append(residue_index)
                            last_residue_number = residue_index
                        else:
                            # Non-contiguous; save the current element and start a new one
                            ss_elements_dict[target_ss].append(current_element)
                            current_element = [residue_index]
                            current_chain = chain_id
                            last_residue_number = residue_index
                else:
                    if current_element:
                        # End of current SS element
                        ss_elements_dict[target_ss].append(current_element)
                        current_element = []
                        current_chain = None
                        last_residue_number = None
            # Append any remaining SS element
            if current_element:
                ss_elements_dict[target_ss].append(current_element)

            # Filter overlapping subsets, keeping only the largest supersets
            ss_elements_dict[target_ss] = self.filter_supersets(ss_elements_dict[target_ss])

        return ss_elements_dict

    def filter_supersets(self, elements):
        """
        Filters a list of lists to keep only the largest supersets in case of overlap.

        Parameters:
            elements (list of list of int): List of lists, where each sublist represents an SS element.

        Returns:
            list of list of int: Filtered list with only the largest supersets.
        """
        elements = sorted(elements, key=len, reverse=True)
        filtered_elements = []

        for elem in elements:
            # Add element only if it is not a subset of any already included element
            if not any(set(elem).issubset(set(super_elem)) for super_elem in filtered_elements):
                filtered_elements.append(elem)

        return filtered_elements


    def process(self) -> None:
        parser = PDBParser(QUIET=True)
        data_list = []

        print(self.raw_file_names)

        for pdb_file in tqdm(self.raw_file_names, desc="Processing PDB files"):
            pdb_path = osp.join(self.pdb_dir, pdb_file)
            structure = parser.get_structure(pdb_file, pdb_path)
            print(pdb_path)
            print(pdb_file)
            print(structure)
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
            dssp_data = []
            for chain in model:
                for residue in chain:
                    if residue.id[0] != ' ':  # Skip hetero residues
                        continue
                    resname = residue.get_resname()
                    residues.append(resname)
                    ss = dssp[(chain.id, residue.id[1])]
                    print(ss)
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

                    # Append to dssp_data for grouping
                    dssp_data.append((chain.id, residue.id, resname, ss_label))
            
            if not residues:
                continue  # Skip if no valid residues

            num_residues = len(residues)
            pos = torch.tensor(coords, dtype=torch.float)
            
            # Group secondary structure elements for all types
            ss_elements = self.group_ss_elements(dssp_data)

            # Encode residue types
            residue_features = self.get_residue_features(residues)
            x = residue_features

            # Create edge_index based on spatial proximity (e.g., within 8 Å)
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
                residues=residues,
                ss_elements=ss_elements,
                mol=0,
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






