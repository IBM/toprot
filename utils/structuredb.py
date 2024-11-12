import os
import shutil
import subprocess
from typing import List, Optional, Tuple

import time
import foldcomp
import pandas as pd
import numpy as np
from tqdm import tqdm
from autopeptideml.data.residues import RESIDUES

RESIDUES_TO_ONE = {v: k for k, v in RESIDUES.items()}


class Protein:
    def __init__(
        self,
        data: pd.DataFrame,
        coords: np.ndarray
    ):
        self.data = data
        self.coords = coords

    @classmethod
    def _read_pdb(self, pdb_txt: str):
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
        return Protein(pd.DataFrame(proto_data), np.array(proto_coords))

    def get_c_alpha(self) -> np.ndarray:
        indxs = self.data['atom_name'] == 'CA'
        return indxs

    def get_sequence(self) -> str:
        indxs = self.get_c_alpha()
        seq = ''.join(self.data[indxs]['residue_name'].tolist())
        return seq


class StructureDB:
    df: pd.DataFrame = pd.DataFrame()

    def __init__(
        self,
        protein_list: List[str] = None,
        data_dir: str = '.',
        db_name: str = 'afdb_swissprot_v4'
    ):
        """
        Initialises an instance of the StructureDB class

        :param df: DataFrame with an `id` with the `UniProt ID`
                   if the database is AlphaFold or the `PDB ID`
                   if the database is the PDB.
                   This DataFrame will be used for obtaining
                   the predictions in the Database.
        :type df: pd.DataFrame
        :param data_dir: _description_, defaults to '.'
        :type data_dir: str, optional
        :param db_name: _description_, defaults to 'afdb_swissprot_v4'
        :type db_name: str, optional
        """
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, db_name)
        self.proteins = protein_list

        if 'afdb' in db_name and self.proteins is not None:
            self.proteins = map(lambda x: f'AF-{x}-F1-model_v4'
                                if 'model' not in x else x,
                                protein_list)
        os.makedirs(data_dir, exist_ok=True)
        self.foldseek = self._check_foldseek()
        self.foldcomp = self._check_foldcomp()

        if not os.path.exists(self.db_path):
            print(f'Downloading Foldcomp DB: {db_name}')
            self._set_up_db(self.data_dir, db_name)

    def _check_foldseek(self) -> str:
        process = subprocess.Popen(['which', 'foldseek'],
                                   stderr=subprocess.PIPE,
                                   stdout=subprocess.PIPE)
        stdout, stderr = process.communicate()
        stdout, stderr = stdout.decode('utf-8'), stderr.decode('utf-8')
        if len(stderr) > 10 and len(stdout) < 1:
            raise ImportError(
                'Please install foldseek:' +
                ' `conda install foldseek -c bioconda`')
        return stdout.strip('\n')

    def _check_foldcomp(self) -> str:
        process = subprocess.Popen(['which', 'foldcomp'],
                                   stderr=subprocess.PIPE,
                                   stdout=subprocess.PIPE)
        stdout, stderr = process.communicate()
        stdout, stderr = stdout.decode('utf-8'), stderr.decode('utf-8')
        if len(stderr) > 10 and len(stdout) < 1:
            raise ImportError(
                'Please install foldcomp: ' +
                '`conda install foldcomp -c bioconda`')
        return stdout.strip('\n')

    def _set_up_db(self, data_dir: str, db: str):
        cwd = os.getcwd()
        data_dir = os.path.abspath(data_dir)
        os.chdir(data_dir)
        foldcomp.setup(db)
        os.chdir(cwd)

    def get_3d_structure(
        self,
        data_dir: Optional[str] = None,
        protein_list: List[str] = None,
    ) -> pd.DataFrame:
        if data_dir is None:

            data_dir = self.data_dir

        if protein_list is not None:
            db = foldcomp.open(self.db_path, ids=protein_list)

        elif self.proteins is not None:
            db = foldcomp.open(self.db_path, ids=self.proteins)

        else:
            db = foldcomp.open(self.db_path)

        output = []
        for (name, pdb) in tqdm(db):
            prot = Protein._read_pdb(pdb)
            output.append(prot)
        return output

    def get_3di_tokens(
        self,
        data_dir: Optional[str] = None,
        foldseek_verbose: bool = False,
        threads: int = 10,
        batch_size: int = 4096
    ) -> pd.DataFrame:
        if data_dir is None:
            data_dir = self.data_dir

        outdir = os.path.join(data_dir, 'structures')
        tmp_db = os.path.join(data_dir, 'structures_tmp')
        tmp_dir = f"{time.time()}"

        if os.path.isdir(outdir):
            shutil.rmtree(outdir)
        if os.path.isdir(tmp_db):
            shutil.rmtree(tmp_db)

        os.makedirs(outdir)
        os.makedirs(tmp_dir)

        data = []
        batch = []

        if self.proteins is not None:
            db = foldcomp.open(self.db_path, ids=self.proteins)

        else:
            db = foldcomp.open(self.db_path)

        for (name, pdb) in tqdm(db):
            # if name.strip('.pdb') not in self.proteins:
            #     continue
            filename = os.path.join(outdir, name)
            with open(filename, 'w') as fo:
                fo.write(pdb)
            batch.append(filename)

            if len(batch) == batch_size:
                batch_data = self._3di_tokens_process_batch(
                    outdir=outdir, tmp_db=tmp_db,
                    tmp_dir=tmp_dir, threads=threads,
                    verbose=foldseek_verbose
                )
                data.extend(batch_data)

                shutil.rmtree(tmp_db)
                shutil.rmtree(outdir)
                os.makedirs(outdir)
                batch = []

        if len(batch) > 0:
            batch_data = self._3di_tokens_process_batch(
                outdir=outdir, tmp_db=tmp_db,
                tmp_dir=tmp_dir, threads=threads,
                verbose=foldseek_verbose
            )
            data.extend(batch_data)
            shutil.rmtree(tmp_db)
            shutil.rmtree(outdir)

        shutil.rmtree(tmp_dir)
        db.close()
        df = pd.DataFrame(data)
        df.id = df.id.map(lambda x: x.split(' ')[1].split('-')[1])

        return df

    def _3di_tokens_process_batch(
        self, outdir: str, tmp_db: str, tmp_dir: str,
        threads: int, verbose: bool
    ) -> List[dict]:
        data = []
        v = '0' if not verbose else '3'
        tmp_save_path = os.path.join(tmp_dir, "tmp.tsv")

        process = subprocess.Popen(
            [self.foldcomp, 'compress', outdir, tmp_db,
                '-t', f"{threads}"], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        stdout = stdout.decode('utf-8')
        stderr = stderr.decode('utf-8')
        if len(stderr) > 1 and len(stdout) < 1:
            raise RuntimeError(f'Foldcomp error: {stderr}')

        subprocess.run([self.foldseek, 'structureto3didescriptor',
                        '-v', v, '--threads', f'{threads}',
                        '--chain-name-mode', '0', tmp_db,
                        tmp_save_path, '--input-format', '5'])

        with open(tmp_save_path, "r") as r:
            for line in r:
                desc, seq, struc_seq = line.split("\t")[:3]
                datum = {
                    'id': desc,
                    'sequence': seq,
                    'structure_tokens': struc_seq.lower()
                }
                data.append(datum)

        os.remove(tmp_save_path)
        os.remove(tmp_save_path + '.dbtype')
        return data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.loc[idx, 'id']

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx > len(self) - 1:
            raise StopIteration()
        item = self.df.iloc[self.idx, :]
        self.idx += 1
        return item


if __name__ == '__main__':
    db = StructureDB(data_dir='db_data', db_name='afdb_swissprot_v4')
    db.get_3d_structure()