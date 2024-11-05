try:
    import openmm.app as app
    from openmm.app.element import hydrogen
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer
except ImportError:
    raise ImportError('To fix PDBs you need to have installed OpenMM and PDBFixer. To do so run: `pip install openmm` and `conda install -c conda-forge pdbfixer`.')


def fix_pdb(self, pdb: str, tmp_path: str) -> str:
    file = open(pdb).readlines()
    header = ""
    for line in file:
        if (line.startswith('TER') or
            line.startswith('HETATM') or
            line.startswith('ANISOU') or
            line.startswith('SSBOND') or
            line.startswith('CONECT') or
            line.startswith('ATOM') or
            line.startswith('MASTER') or
           line.startswith('END')):
            continue
        header += line

    try:
        fixer = PDBFixer(tmp_path)
        modeller = app.Modeller(fixer.topology, fixer.positions)
        toDelete = [atom for atom in modeller.topology.atoms()
                    if atom.element == hydrogen]
        modeller.delete(toDelete)
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions
        fixer.removeHeterogens(False)

        with open(tmp_path, 'w') as file:
            file.write(header)
            PDBFile.writeFile(fixer.topology, fixer.positions, file)

    except AttributeError as e:
        print(e)
        pass

    crysted = False
    file = open(tmp_path).readlines()

    with open(tmp_path, 'w') as fo:
        for line in file:
            if (line.startswith('REMARK   1 CREATED WITH OPENMM 8.1.1,') or
                line.startswith('HELIX') or
                line.startswith('SHEET') or
               line.startswith('REMARK')):
                continue
            elif line.startswith('CRYST1') and crysted:
                continue
            else:
                if line.startswith('CRYST1'):
                    crysted = True
                fo.write(line)
    return tmp_path
