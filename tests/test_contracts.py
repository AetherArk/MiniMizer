"""JSON boundaries; mocked failures never stand in for scientific smoke tests."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from jsonschema import Draft202012Validator
import minimize
from minimizer import contracts

PACKET = Path(__file__).resolve().parents[1] / 'example'


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.config = contracts.load(PACKET / 'config.json', 'config')

    def test_schemas_are_valid(self):
        for name in ('config', 'report', 'atoms'):
            Draft202012Validator.check_schema(json.loads(
                (contracts.SCHEMAS / f'{name}.schema.json').read_text()))

    def test_required_config_fields(self):
        for key in self.config:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.config)
                del bad[key]
                with self.assertRaises(ValueError):
                    contracts.validate(bad, 'config')

    def test_invalid_config_never_executes(self):
        mutations = [
            ('schema_version', '99'), ('protocol', '99'), ('source_smiles', 1),
            ('fixed_ligand_source_indices', [True]),
            ('fixed_ligand_source_indices', [21.0]),
            ('fixed_ligand_source_indices', [-1]),
            ('fixed_ligand_source_indices', [21, 21]),
            ('fixed_ligand_source_indices', [999]),
            ('ligand_state', dict(self.config['ligand']['ligand_state'], atom_count=1)),
            ('mapping', dict(self.config['ligand']['mapping'], injected='unexpected')),
            ('unknown', '$(touch should-not-exist)'),
        ]
        mutations += [('unknown', x) for x in (float('nan'), float('inf'), -float('inf'))]
        for key, value in mutations:
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as td:
                packet, out = Path(td) / 'input', Path(td) / 'output'
                packet.mkdir()
                for src in PACKET.iterdir():
                    if src.name != 'config.json':
                        (packet / src.name).symlink_to(src)
                bad = copy.deepcopy(self.config)
                (bad['ligand'] if key in ('source_smiles', 'mapping', 'ligand_state', 'fixed_ligand_source_indices') else bad)[key] = value
                (packet / 'config.json').write_text(json.dumps(bad))
                with patch('sys.argv', ['minimize.py', str(packet / 'config.json'), str(out)]), \
                        patch('minimize.stage1') as stage, patch('minimize.subprocess.run') as run:
                    with self.assertRaises(ValueError):
                        minimize.main()
                    stage.assert_not_called()
                    run.assert_not_called()
                    self.assertFalse(out.exists())

    def test_nested_atom_contract(self):
        for key, value in [('package_serial', '4035'), ('explicit_H', -1),
                           ('source_graph_atom_index', False), ('extra', 0)]:
            bad = copy.deepcopy(self.config)
            bad['ligand']['mapping']['package_graphs']['L']['atoms'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                contracts.validate(bad, 'config')

    def test_invalid_json_and_duplicate_fields(self):
        for text in ('{', '{"protocol":"0.1","protocol":"0.1"}'):
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'config.json'
                path.write_text(text)
                with self.assertRaises(ValueError):
                    contracts.load(path, 'config')

    def test_failure_has_no_fake_metrics(self):
        failure = dict(schema_version='1.0', protocol='0.1', validation_pass=False,
                       stage1_arm=None, error='No stereo-valid ligand minimum')
        contracts.validate(failure, 'report')
        for key, value in [('validation_pass', True), ('schema_version', '2'),
                           ('optimization', None), ('error', ''), ('extra', 0)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                contracts.validate(dict(failure, **{key: value}), 'report')

    def test_bad_worker_output_cannot_report_success(self):
        for worker in ({'validation_pass': True}, {'optimization': float('nan')}, []):
            with self.subTest(worker=worker), tempfile.TemporaryDirectory() as td:
                out = Path(td) / 'output'
                def worker_run(*args, **kwargs):
                    (out / 'report.json').write_text(json.dumps(worker))
                with patch('sys.argv', ['minimize.py', str(PACKET / 'config.json'), str(out)]), \
                        patch('minimize.stage1', return_value='test-arm'), \
                        patch('minimize.subprocess.run', side_effect=worker_run), \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        minimize.main()
                self.assertEqual(caught.exception.code, 1)
                report = contracts.load(out / 'report.json', 'report')
                self.assertFalse(report['validation_pass'])
                self.assertIn('error', report)
                self.assertNotIn('optimization', report)

    def test_stage1_failure_is_serializable(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'output'
            with patch('sys.argv', ['minimize.py', str(PACKET / 'config.json'), str(out)]), \
                    patch('minimize.stage1', side_effect=ValueError('No valid minimum')), \
                    patch('minimize.subprocess.run') as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    minimize.main()
                self.assertEqual(run.call_count, 1)  # OpenMM preflight only
                self.assertIn('--validate-only', run.call_args.args[0])
            report = contracts.load(out / 'report.json', 'report')
            self.assertEqual(report['error'], 'ValueError: No valid minimum')
