import os
import os.path as osp
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from autopeptideml.data.residues import RESIDUES
from .ss_process import DSSP
from .pocket_process import FPocket

RESIDUES_TO_ONE = {v: k for k, v in RESIDUES.items()}


class Protein:
    def __init__(
        self,
        data: pd.DataFrame,
        coords: np.ndarray,
        text: Optional[str] = None
    ):
        self.data = data
        self.coords = coords
        self.text = text
        self.atom_features = None
        self.res_features = None
        self.ss = None
        self.ss_features = None
        self.pockets = None
        self.pocket_features = None
        self.domains = None

    @classmethod
    def _read_pdb(self, pdb_txt: str, save_text: Optional[bool] = True):
        proto_data = []
        proto_coords = []
        for line in pdb_txt.split('\n'):
            if not line.startswith('ATOM'):
                continue
            if line[76:78].strip() == 'H':
                continue
            proto_data.append(
                {
                    'atom_number': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'residue_name': RESIDUES_TO_ONE[line[17:20]],
                    'residue_number': int(line[22:26]),
                    'chain_id': line[21:22].strip(),
                    'element_symbol': line[76:78].strip()
                }
            )
            proto_coords.append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
        if save_text:
            return Protein(pd.DataFrame(proto_data), np.array(proto_coords),
                           pdb_txt)
        else:
            return Protein(pd.DataFrame(proto_data), np.array(proto_coords))

    def get_c_alpha(self) -> np.ndarray:
        indxs = self.data['atom_name'] == 'CA'
        return indxs

    def get_sequence(self) -> str:
        indxs = self.get_c_alpha()
        seq = ''.join(self.data[indxs]['residue_name'].tolist())
        return seq

    def get_ss(self, dssp: Optional[DSSP] = None,
               dictionary: Optional[int] = 8,
               tmp_dir: Optional[str] = 'tmp') -> np.ndarray:
        if self.text is None:
            raise RuntimeError('Protein object has to be initialised with `save_text` option.')

        if dssp is None:
            dssp = DSSP()
        if not osp.exists(tmp_dir):
            os.mkdir(tmp_dir)
        tmp_name = f'{time.time()}.pdb'
        pdb_path = osp.join(tmp_dir, tmp_name)
        with open(pdb_path, 'w') as fo:
            fo.write(self.text)
        ss = dssp.get_ss(pdb_path, dictionary)
        self.ss = ss
        os.remove(pdb_path)
        return np.array(ss).astype(np.int8)

    def get_pockets(self, fpocket: Optional[FPocket] = None,
                    threshold: Optional[float] = 0.0,
                    k: Optional[int] = None,
                    tmp_dir: Optional[str] = 'tmp') -> Tuple[List[np.ndarray], np.ndarray]:
        df = self.data[self.get_c_alpha()]

        def _get_res_index(res) -> int:
            return df[(df['residue_name'] == res[0])
                      & (df['residue_number'] == res[1])
                      & (df['chain_id'] == res[2])].index.to_list()[0]

        if self.text is None:
            raise RuntimeError('Protein object has to be initialised with `save_text` option.')

        if fpocket is None:
            fpocket = FPocket()
        if not osp.exists(tmp_dir):
            os.mkdir(tmp_dir)
        tmp_name = f'{time.time()}.pdb'
        pdb_path = osp.join(tmp_dir, tmp_name)
        with open(pdb_path, 'w') as fo:
            fo.write(self.text)
        pckt_res, pckt_props = fpocket.get_pockets(
            pdb_path, tmp_dir=tmp_dir, threshold=threshold
        )
        if pckt_res is not None:
            pckt_res = [
                np.array([_get_res_index(
                    (RESIDUES_TO_ONE[res[0]], res[1], res[2])
                ) for res in pocket]) for pocket in pckt_res]
            self.pockets = pckt_res
            self.pocket_features = pckt_props

        return pckt_res, pckt_props
