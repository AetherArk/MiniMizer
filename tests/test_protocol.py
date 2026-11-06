"""Prepared-input checks; coordinate perturbations are functional tests, not datasets."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
from rdkit import Chem
import minimize
from minimizer import ligand, stereo

PACKET = Path(__file__).resolve().parents[1] / 'example'


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = minimize.validate(PACKET)
        cls.mapping = cls.config['mapping']
        cls.mol, _, _ = ligand.pdb_binding(PACKET / 'input.pdb', cls.mapping, cls.config['source_smiles'])

    def test_fresh_coordinates_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / 'packet'
            shutil.copytree(PACKET, d)
            _, writer, _ = ligand.pdb_binding(d / 'input.pdb', self.mapping, self.config['source_smiles'])
            xyz = self.mol.GetConformer().GetPositions().copy()
            xyz[0, 0] += .01
            (d / 'input.pdb').write_bytes(writer(xyz)[0])
            minimize.validate(d)

    def test_missing_forcefield_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / 'packet'
            shutil.copytree(PACKET, d)
            (d / 'system.xml.gz').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing prepared input'):
                minimize.validate(d)

    def test_receptor_order_rejected(self):
        import gzip
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / 'packet'
            shutil.copytree(PACKET, d)
            meta = json.loads(gzip.decompress((d / 'atoms.json.gz').read_bytes()))
            meta[0], meta[1] = meta[1], meta[0]
            (d / 'atoms.json.gz').write_bytes(gzip.compress(json.dumps(meta).encode()))
            with self.assertRaisesRegex(ValueError, 'ordering'):
                minimize.validate(d)

    @unittest.skipUnless(os.environ.get('MINIMIZER_MD_IMAGE'), 'set MINIMIZER_MD_IMAGE for offline integration')
    def test_perturbed_coordinates_end_to_end(self):
        # Deliberate functional perturbation; same chemical graph and frozen parameters.
        with tempfile.TemporaryDirectory() as td:
            d, out = Path(td) / 'input', Path(td) / 'output'
            shutil.copytree(PACKET, d)
            _, writer, _ = ligand.pdb_binding(d / 'input.pdb', self.mapping, self.config['source_smiles'])
            xyz = self.mol.GetConformer().GetPositions().copy()
            xyz[0, 0] += .01
            (d / 'input.pdb').write_bytes(writer(xyz)[0])
            subprocess.run([sys.executable, '-B', str(PACKET.parent / 'minimize.py'), str(d), str(out),
                            '--md-image', os.environ['MINIMIZER_MD_IMAGE']], check=True)
            report = json.loads((out / 'report.json').read_text())
            self.assertTrue(report['validation_pass'])
            self.assertTrue(report['optimization']['scipy_success'])
            self.assertTrue((out / report['candidate']).is_file())
            x0 = np.load(out / 'starting_positions.npy')
            x = np.load(out / 'local/k12552/positions.npy')
            mobility = json.loads((out / 'mobility.json').read_text())
            fixed = sorted(set(range(len(x))) - set(mobility['movable']))
            np.testing.assert_array_equal(x[fixed], x0[fixed])
            # Full-system restraint references must be this run's coordinates.
            import xml.etree.ElementTree as ET
            root = ET.parse(out / 'system.xml').getroot()
            for force in root.findall('Forces/Force'):
                if force.get('type') == 'CustomExternalForce':
                    particles = force.findall('Particles/Particle')
                    self.assertTrue(particles)
                    for atom in particles:
                        ref = [float(atom.attrib['param'+str(i)]) for i in (1, 2, 3)]
                        np.testing.assert_array_equal(ref, x0[int(atom.attrib['index'])])

    def test_mapping_state_mismatch_rejected(self):
        bad = copy.deepcopy(self.mapping)
        bad['package_graphs']['L']['exact_source_smiles'] += 'C'
        with self.assertRaisesRegex(ValueError, 'state_mismatch'):
            ligand.pdb_binding(PACKET / 'input.pdb', bad, self.config['source_smiles'])

    def test_ligand_graph_and_identity_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'input.pdb'
            original = (PACKET / 'input.pdb').read_text()
            path.write_text(''.join(line for line in original.splitlines(keepends=True)
                                    if not line.startswith('CONECT')))
            with self.assertRaisesRegex(ValueError, 'graph_mismatch'):
                ligand.pdb_binding(path, self.mapping, self.config['source_smiles'])
            bad = copy.deepcopy(self.mapping)
            bad['package_graphs']['L']['atoms'][0]['package_atom_name'] = 'WRONG'
            with self.assertRaisesRegex(ValueError, 'identity_mismatch'):
                ligand.pdb_binding(PACKET / 'input.pdb', bad, self.config['source_smiles'])

    def test_real_mmff_gradient(self):
        h, plans = stereo.plans(self.mol)
        x, fun = stereo.objective(self.mol, h, plans[0], 300., 1000.)
        flat = x.ravel()
        _, gradient = fun(flat)
        # Finite differences of three components; no minimization or new molecule.
        for i in (0, 7, len(flat)-1):
            plus, minus = flat.copy(), flat.copy()
            plus[i] += 1e-6
            minus[i] -= 1e-6
            fd = (fun(plus)[0]-fun(minus)[0])/2e-6
            self.assertAlmostEqual(fd, gradient[i], delta=1e-4)

    def test_anchor_and_input_are_unchanged(self):
        h, plans = stereo.plans(self.mol)
        before = self.mol.GetConformer().GetPositions().copy()
        ligand.ACTIVE_ANCHORS = [21]
        moved, _, report = ligand.anchored_optimize(self.mol, h, plans[0], 300., 1000.)
        np.testing.assert_array_equal(before, self.mol.GetConformer().GetPositions())
        np.testing.assert_array_equal(before[21], moved.GetConformer().GetPositions()[21])
        self.assertTrue(report['converged'])

    def test_historical_selection_gates(self):
        a = dict(converged=True, tetrahedra={'valid': True}, rounded_tetrahedra={'valid': True},
                 full_H_stereo=True, rounded_full_H_stereo=True, heavy_stereo=True,
                 strict_graph=True, strict_coordinate_stereo=True, rmsd_A=.5, max_movement_A=1.)
        self.assertTrue(minimize.eligible(a))
        for key, val in [('rmsd_A', .50001), ('max_movement_A', 1.00001),
                         ('converged', False), ('strict_graph', False), ('heavy_stereo', False)]:
            self.assertFalse(minimize.eligible(dict(a, **{key: val})))
        # PoseBusters is not a Stage 1 prefilter in this historical worker.
        self.assertTrue(minimize.eligible(dict(a, posebusters_pass=False)))

    def test_real_pdb_roundtrip_keeps_nonligand_bytes(self):
        _, writer, targets = ligand.pdb_binding(PACKET / 'input.pdb', self.mapping, self.config['source_smiles'])
        original = (PACKET / 'input.pdb').read_bytes().splitlines(keepends=True)
        changed, _ = writer(self.mol.GetConformer().GetPositions() + .01)
        for i, (a, b) in enumerate(zip(original, changed.splitlines(keepends=True))):
            if i not in targets:
                self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
