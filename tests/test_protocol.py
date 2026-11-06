"""Focused tests on the real packet; no generated molecular fixtures."""
import copy
import json
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
        cls.manifest = minimize.validate(PACKET)
        cls.mapping = json.loads((PACKET / 'mapping.json').read_text())
        cls.mol, _, _ = ligand.pdb_binding(PACKET / 'input.pdb', cls.mapping, cls.manifest['source_smiles'])

    def test_packet_integrity_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / 'packet'
            shutil.copytree(PACKET, d)
            (d / 'input.pdb').write_bytes((d / 'input.pdb').read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'identity'):
                minimize.validate(d)

    def test_missing_hash_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            manifest = copy.deepcopy(self.manifest)
            del manifest['files']['system.xml.gz']
            (d / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                minimize.validate(d)

    def test_mapping_state_mismatch_rejected(self):
        bad = copy.deepcopy(self.mapping)
        bad['package_graphs']['L']['exact_source_smiles'] += 'C'
        with self.assertRaisesRegex(ValueError, 'state_mismatch'):
            ligand.pdb_binding(PACKET / 'input.pdb', bad, self.manifest['source_smiles'])

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
        _, writer, targets = ligand.pdb_binding(PACKET / 'input.pdb', self.mapping, self.manifest['source_smiles'])
        original = (PACKET / 'input.pdb').read_bytes().splitlines(keepends=True)
        changed, _ = writer(self.mol.GetConformer().GetPositions() + .01)
        for i, (a, b) in enumerate(zip(original, changed.splitlines(keepends=True))):
            if i not in targets:
                self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
