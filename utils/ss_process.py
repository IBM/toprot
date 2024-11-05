import os
import subprocess as sbp
from typing import List, Optional, Tuple


DSSP_VOCAB = ['H', 'B', 'E', 'G', 'I', 'P', 'T', 'S', ' ']
DSSP_3CODE = {'Helix': 0, 'Sheet': 1, 'Loop': 2}
CODE8_TO_3CODE = {'H': 0, 'B': 1, 'E': 1,
                  'G': 0, 'I': 0, 'P': 0,
                  'T': 2, 'S': 2, ' ': 2}
DSSP_8CODE = {letter: i for i, letter in enumerate(DSSP_VOCAB)}


class DSSP:
    def __init__(self, dssp_path: Optional[str] = 'mkdssp'):
        self.program = dssp_path

        dssp = sbp.Popen(['which', dssp_path], stdout=sbp.PIPE,
                         stderr=sbp.PIPE)
        dssp = dssp.communicate()[0].decode('utf-8').strip()
        if not os.path.exists(dssp):
            raise ImportError('Please install the dssp program. ' +
                              '`conda install sbl::dssp`')

    def _read_dssp_output(self, output: str) -> Tuple[List[int], List[int]]:
        ss3, ss8 = [], []
        started = False
        # diccionary = CODE8_TO_3CODE if num_classes == 3 else DSSP_8CODE

        for line in output.split('\n'):
            if line.startswith('  #'):
                started = True
                continue
            if not started or line == '':
                continue
            if '!' in line:
                continue
            ss3.append(CODE8_TO_3CODE[line[16]])
            ss8.append(DSSP_8CODE[line[16]])
        return ss3, ss8

    def get_ss(self, pdb_path: str, ss_dict: Optional[int] = 8) -> List[int]:
        """Get Secondary Structure labels for all residues in PDB at `pdb_path`.

        :param pdb_path: Path where the PDB is located
        :type pdb_path: str
        :param ss_dict: Secondary Structure dictionary.
            Options:
                - 3: alpha-helix, beta-sheet, or coil
                - 8: alpha-helix, beta-bridge, beta-ladder, 3_10-helix, pi-helix, turn, bend, coil
            Defaults to 8
        :type ss_dict: Optional[int], optional
        :return: List of labels for each residue
        :rtype: List[int]
        """
        if ss_dict not in [3, 8]:
            raise ValueError(f'SS dictionary can only be 3 or 8. Value provided: {ss_dict}')
        pdb_path = os.path.abspath(pdb_path)
        dssp_output = sbp.Popen(
            [self.program, pdb_path, '--output-format',
             'dssp', '--write-other', ''],
            stdout=sbp.PIPE, stderr=sbp.PIPE
        )
        stdout, stderr = dssp_output.communicate()
        if len(stderr) > 0 and len(stdout) < 10:
            raise RuntimeError(stderr.decode('utf-8'))

        ss3, ss8 = self._read_dssp_output(stdout.decode('utf-8'))
        if ss_dict == 3:
            return ss3
        elif ss_dict == 8:
            return ss8


if __name__ == '__main__':
    dssp = DSSP()
    dssp.get_ss('../1dhy.pdb')
