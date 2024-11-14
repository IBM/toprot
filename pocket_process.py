import os
import os.path as osp
import shutil
import subprocess as sbp
from typing import List, Optional

from biopandas.pdb.pandas_pdb import PandasPdb
import numpy as np
import pandas as pd


def _get_pockets(output_path: str) -> pd.DataFrame:
    data = []
    with open(output_path) as fi:
        for line in fi:
            line = line.strip('\n')
            if not line.startswith('ATOM'):
                continue
            entry = {
                'pocket': int(line[24:29]),
                'center_x': float(line[30:38]),
                'center_y': float(line[39:46]),
                'center_z': float(line[47:54])
            }
            data.append(entry)
    return pd.DataFrame(data)


def _get_residues(outputdir: str, df: pd.DataFrame) -> pd.DataFrame:
    pockets = df.pocket.unique()
    data = []
    for pocket in pockets:
        path = os.path.join(outputdir, f'pocket{pocket}_atm.pdb')
        pdb_df = PandasPdb().read_pdb(path)
        residues = pdb_df.df['ATOM']['residue_number']
        chains = pdb_df.df['ATOM']['chain_id']
        resnames = pdb_df.df['ATOM']['residue_name']
        residue_list = list(set([(name, number, chain)
                                 for name, number, chain in
                                 zip(resnames, residues, chains)]))
        data.append(residue_list)
    # df['residue_list'] = data
    return data


def _get_centers(df: pd.DataFrame) -> pd.DataFrame:
    pockets = df.pocket.unique()
    data = []
    for pocket in pockets:
        tmp_df = df[df.pocket == pocket]
        entry = {
            'pocket': pocket,
            'center_x': tmp_df.center_x.mean(),
            'center_y': tmp_df.center_y.mean(),
            'center_z': tmp_df.center_z.mean()
        }
        data.append(entry)
    return pd.DataFrame(data)


def _get_properties(scores_path: str, df: pd.DataFrame) -> pd.DataFrame:
    PROPERTIES = [
        'Score', 'Druggability Score', 'Number of Alpha Spheres',
        'Total SASA', 'Polar SASA', 'Apolar SASA', 'Volume',
        'Mean local hydrophobic density', 'Mean alpha sphere radius',
        'Mean alp. sph. solvent access', 'Apolar alpha sphere proportion',
        'Hydrophobicity score', 'Volume score', 'Polarity score',
        'Charge score', 'Proportion of polar atoms', 'Alpha sphere density',
        'Cent. of mass - Alpha Sphere max dist', 'Flexibility'
    ]
    with open(scores_path) as fi:
        for line in fi:
            line = line.strip('\n')
            if line.startswith('Pocket '):
                last_pocket = int(line.split(' ')[1])
                continue
            for property in PROPERTIES:
                if property in line:
                    value = float(line.split(':')[1])
                    df.loc[df.pocket == last_pocket, property] = value
    return df


class FPocket:
    def __init__(self, fpocket_path: Optional[str] = 'fpocket'):
        self.program = fpocket_path
        dssp = sbp.Popen(['which', fpocket_path], stdout=sbp.PIPE,
                         stderr=sbp.PIPE)
        dssp = dssp.communicate()[0].decode('utf-8').strip()
        if not osp.exists(dssp):
            raise ImportError('Please install the Fpocket program. ' +
                              '`conda install fpocket -c conda-forge`')

    def get_pockets(self, pdb_path: str, pdb_df: PandasPdb, tmp_dir: str,
                    threshold: Optional[float] = 0.) -> List[np.ndarray]:
        pdb_path = osp.abspath(pdb_path)
        new_pdb_path = osp.join(
            tmp_dir, osp.basename(pdb_path)
        )
        pdb_id = osp.basename(pdb_path).strip('.pdb')
        out_dir = osp.abspath(osp.join(
            tmp_dir, osp.basename(pdb_path).strip('.pdb') + "_out"
        ))
        shutil.copyfile(pdb_path, new_pdb_path)
        process = sbp.Popen(
            [self.program, '-f', new_pdb_path],
            stdout=sbp.PIPE, stderr=sbp.PIPE
        )
        out, err = process.communicate()
        if len(err) > 1 and len(out) == 0:
            raise RuntimeError(f"Fpocket error: {err.decode('utf-8')}")

        out = out.decode('utf-8')
        pockets_path = os.path.join(out_dir, f'{pdb_id}_pockets.pqr')
        residues_path = os.path.join(out_dir, 'pockets')
        scores_path = os.path.join(out_dir, f'{pdb_id}_info.txt')

        df = _get_pockets(pockets_path)
        df = _get_centers(df)
        df = _get_properties(scores_path, df)
        df = df[df.Score > threshold].reset_index(drop=True)
        df = df.sort_values(by='Score', ascending=False)
        pocket_properties = {
            int(r.pocket): r.iloc[4:].tolist() for _, r in df.iterrows()
        }
        pocket_residues = _get_residues(residues_path, df)
        pdb_df = pdb_df[pdb_df["atom_name"] == 'CA']
        output_res = []
        for _, res_df in pdb_df.iterrows():
            res_pockets = []
            for pocket, residues in enumerate(pocket_residues):
                for res in residues:
                    if ((res_df.residue_name == res[0]) &
                            (res_df.residue_number == res[1]) &
                            (res_df.chain_id == res[2])):
                        res_pockets.append(pocket)
            if len(res_pockets) == 1:
                output_res.append(res_pockets[0])
            elif len(res_pockets) > 1:
                output_res.append(res_pockets)
            else:
                output_res.append(-1)

        shutil.rmtree(out_dir)
        return output_res, pocket_properties


if __name__ == '__main__':
    fp = FPocket()
    pdb_df = PandasPdb().read_pdb('../1dhy.pdb').df['ATOM']
    fp.get_pockets('../1dhy.pdb', pdb_df, '../tmp'