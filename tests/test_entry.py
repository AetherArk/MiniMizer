"""Single-file entry: dependency relocation and fail-before-optimization checks."""
import contextlib
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import minimize
from minimizer import contracts

EXAMPLE = Path(__file__).resolve().parents[1] / 'example'


def relocate(root):
    config = contracts.load(EXAMPLE / 'config.json', 'config')
    entry = root / 'request' / 'job.json'
    entry.parent.mkdir()
    for i, ref in enumerate(minimize.references(config)):
        target = root / f'dependency-{i}' / f'renamed-{i}.data'
        target.parent.mkdir()
        shutil.copyfile(EXAMPLE / ref['path'], target)
        ref['path'] = os.path.relpath(target, entry.parent)
    entry.write_text(json.dumps(config))
    return entry


class EntryTests(unittest.TestCase):
    def test_relocated_references(self):
        with tempfile.TemporaryDirectory() as td:
            config = minimize.validate(relocate(Path(td)))
            for ref in minimize.references(config):
                self.assertTrue(Path(ref['path']).is_absolute())
                self.assertTrue(Path(ref['path']).is_file())

    def test_bad_references_and_units_never_optimize(self):
        for which in range(4):
            for field, value in [('path', 'does-not-exist'), ('format', 'unsupported')]:
                with self.subTest(which=which, field=field), tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    entry = relocate(root)
                    config = json.loads(entry.read_text())
                    minimize.references(config)[which][field] = value
                    entry.write_text(json.dumps(config))
                    with patch('sys.argv', ['minimize.py', str(entry), str(root / 'out')]), \
                            patch('minimize.stage1') as stage, patch('minimize.subprocess.run') as run:
                        with self.assertRaises(ValueError):
                            minimize.main()
                        stage.assert_not_called()
                        run.assert_not_called()
        config = contracts.load(EXAMPLE / 'config.json', 'config')
        config['receptor']['coordinates']['units'] = 'angstrom'
        with self.assertRaises(ValueError):
            contracts.validate(config, 'config')

    def test_required_nested_fields_and_no_overrides(self):
        config = contracts.load(EXAMPLE / 'config.json', 'config')
        for section in ('ligand', 'receptor', 'forcefield'):
            for key in config[section]:
                bad = json.loads(json.dumps(config))
                del bad[section][key]
                with self.subTest(section=section, missing=key), self.assertRaises(ValueError):
                    contracts.validate(bad, 'config')
            bad = json.loads(json.dumps(config))
            bad[section]['maxiter'] = 12
            with self.subTest(section=section, override=True), self.assertRaises(ValueError):
                contracts.validate(bad, 'config')
        config['schema_version'] = '1.0'
        with self.assertRaises(ValueError):
            contracts.validate(config, 'config')

    def test_corrupt_dependencies_never_optimize(self):
        for which, content in [(2, gzip.compress(b'[{"index":0}]')),
                               (3, gzip.compress(b'<System type="System"><Particles/></System>'))]:
            with self.subTest(which=which), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                entry = relocate(root)
                config = json.loads(entry.read_text())
                (entry.parent / minimize.references(config)[which]['path']).write_bytes(content)
                with patch('sys.argv', ['minimize.py', str(entry), str(root / 'out')]), \
                        patch('minimize.stage1') as stage, patch('minimize.subprocess.run') as run:
                    with self.assertRaises(ValueError):
                        minimize.main()
                    stage.assert_not_called()
                    run.assert_not_called()

    def test_openmm_preflight_failure_never_optimizes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            entry = relocate(root)
            with patch('sys.argv', ['minimize.py', str(entry), str(root / 'out')]), \
                    patch('minimize.stage1') as stage, \
                    patch('minimize.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'preflight')), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    minimize.main()
                stage.assert_not_called()
                self.assertFalse(contracts.load(root / 'out/report.json', 'report')['validation_pass'])

    @unittest.skipUnless(os.environ.get('MINIMIZER_MD_IMAGE'), 'set MINIMIZER_MD_IMAGE for offline integration')
    def test_relocated_entry_from_other_cwd_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            entry = relocate(root)
            cwd = root / 'unrelated-cwd'
            cwd.mkdir()
            out = root / 'result'
            subprocess.run([sys.executable, '-B', str(EXAMPLE.parent / 'minimize.py'),
                            str(entry), str(out), '--md-image', os.environ['MINIMIZER_MD_IMAGE']],
                           cwd=cwd, check=True)
            report = contracts.load(out / 'report.json', 'report')
            self.assertTrue(report['validation_pass'])
            self.assertTrue((out / report['candidate']).is_file())
            self.assertEqual(report['optimization']['fixed_max_movement_nm'], 0.)
            staged = contracts.load(out / 'inputs/entry.json', 'config')
            for ref in minimize.references(staged):
                self.assertTrue((out / 'inputs' / ref['path']).is_file())
                self.assertNotIn(str(root), ref['path'])
