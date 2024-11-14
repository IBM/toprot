from torch_geometric.loader import DataLoader
import etnn.prot
from etnn.prot.protcc import ProtCC

# Define parameters
root = './data/proteins'
pdb_dir = './data/proteins/raw'  
lifters = ['residue:0', 'secondary_structure:1']
neighbor_types = ['spatial']  
connectivity = 'self' 

# Initialize dataset
dataset = ProtCC(
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
   

