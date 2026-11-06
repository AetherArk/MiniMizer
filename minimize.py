#!/usr/bin/env python3
"""Minimize ligand coordinates using prepared, atom-mapped force-field inputs."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENMM_CPU_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from rdkit import Chem
from minimizer import stereo as mm
from minimizer import ligand, contracts


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def references(config):
    return (config['ligand']['starting_complex'], config['receptor']['coordinates'],
            config['receptor']['atom_metadata'], config['forcefield']['system'])


def validate(entry):
    config = contracts.load(entry, 'config')
    for ref in references(config):
        path = (entry.parent / ref['path']).resolve()
        if not path.is_file():
            raise ValueError('Missing prepared input: ' + str(path))
        ref['path'] = str(path)
    receptor = np.load(config['receptor']['coordinates']['path'], allow_pickle=False)
    meta = contracts.validate(contracts.parse_json(gzip.decompress(
        Path(config['receptor']['atom_metadata']['path']).read_bytes()).decode(), 'atoms'), 'atoms')
    if receptor.shape != (len(meta), 3) or not np.isfinite(receptor).all():
        raise ValueError('Invalid receptor coordinates')
    if [a['index'] for a in meta] != list(range(len(meta))):
        raise ValueError('Invalid receptor atom ordering')
    identities = {(a['chain'], a['resid'], a['name']): a for a in meta}
    if len(identities) != len(meta):
        raise ValueError('Duplicate receptor identity')
    seen = set()
    for line in (Path(config['ligand']['starting_complex']['path'])).read_text().splitlines():
        if not line.startswith(('ATOM  ', 'HETATM')) or line[21] == 'L':
            continue
        name = line[12:16].strip()
        if line[21] == 'H':
            name = config['receptor']['heme_pdb_names'].get(name, name)
        key = (line[21], line[22:26].strip(), name)
        a = identities.get(key)
        if a is None or key in seen or line[76:78].strip().upper() != a['element'].upper():
            raise ValueError('Receptor identity mismatch')
        seen.add(key)
        xyz = np.array([float(line[k:k+8]) for k in (30, 38, 46)])
        if not np.allclose(xyz, receptor[a['index']]*10, atol=.00051, rtol=0):
            raise ValueError('PDB receptor differs from prepared receptor')
    heavy = {key for key, a in identities.items() if a['element'] != 'H'}
    if not heavy <= seen:
        raise ValueError('Missing receptor heavy atoms')
    mol, _, _ = ligand.pdb_binding(Path(config['ligand']['starting_complex']['path']), config['ligand']['mapping'], config['ligand']['source_smiles'])
    if any(i >= mol.GetNumAtoms() for i in config['ligand']['fixed_ligand_source_indices']):
        raise ValueError('Invalid axial indices')
    full = Chem.AddHs(Chem.Mol(mol))
    state = config['ligand']['ligand_state']
    if (Chem.GetFormalCharge(full) != state['source_formal_charge']
            or full.GetNumAtoms() != state['atom_count']
            or sum(a.GetAtomicNum() == 1 for a in full.GetAtoms()) != state['H_count']):
        raise ValueError('Ligand force-field state mismatch')
    # Structural validation before any optimization, without requiring host OpenMM.
    import xml.etree.ElementTree as ET
    root = ET.fromstring(gzip.decompress(Path(config['forcefield']['system']['path']).read_bytes()))
    if root.tag != 'System' or root.get('type') != 'System':
        raise ValueError('Invalid force-field System XML')
    particles = root.find('Particles')
    if particles is None or len(particles) != len(meta) + full.GetNumAtoms():
        raise ValueError('Force-field particle count mismatch')
    for node in root.iter():
        for value in node.attrib.values():
            try:
                number = float(value)
            except ValueError:
                continue
            if not np.isfinite(number):
                raise ValueError('Nonfinite force-field parameter')
    forces = root.findall('Forces/Force')
    external = [f for f in forces if f.get('type') == 'CustomExternalForce']
    if len(external) != 1 or external[0].get('energy') != '0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)':
        raise ValueError('Unsupported external force')
    if sum(f.get('type') == 'NonbondedForce' for f in forces) != 1 or sum(
            f.get('type') == 'CustomNonbondedForce' for f in forces) != 2:
        raise ValueError('Unsupported prepared force-field model')
    return config


def eligible(a):
    return (a['converged'] and a['tetrahedra']['valid'] and a['rounded_tetrahedra']['valid']
            and a['full_H_stereo'] and a['rounded_full_H_stereo'] and a['heavy_stereo']
            and a['strict_graph'] and a['strict_coordinate_stereo']
            and a['rmsd_A'] <= .5 and a['max_movement_A'] <= 1.)


def stage1(out, config):
    mapping = config['ligand']['mapping']
    m, writer, _ = ligand.pdb_binding(Path(config['ligand']['starting_complex']['path']), mapping, config['ligand']['source_smiles'])
    original = m.GetConformer().GetPositions().copy()
    anchors = config['ligand']['fixed_ligand_source_indices']
    if any(i < 0 or i >= len(original) for i in anchors) or len(set(anchors)) != len(anchors):
        raise ValueError('Invalid axial indices')
    ligand.ACTIVE_ANCHORS = anchors
    h, plans = mm.plans(m)
    attempts, candidates = [], {}
    for ks in mm.DEFAULT['stereo_k']:
        for k in mm.DEFAULT['position_k']:
            for j, plan in enumerate(plans):
                label = f'k{k:g}_ks{ks:g}_orientation{j}'
                moved, full, report = ligand.anchored_optimize(m, h, plan, k, ks)
                data, rounded = writer(moved.GetConformer().GetPositions())
                # Parse serialized coordinates through the same graph-bound reader.
                tmp = out / 'stage1.pdb'
                tmp.write_bytes(data)
                rt, _, _ = ligand.pdb_binding(tmp, mapping, config['ligand']['source_smiles'])
                if not np.array_equal(rt.GetConformer().GetPositions(), rounded):
                    raise ValueError('PDB round-trip mismatch')
                fullx = full.GetConformer().GetPositions().copy()
                fullx[:len(original)] = rounded
                rh = mm.atcoords(full, fullx)
                delta = rounded - original
                report.update(arm=label, rounded_tetrahedra=mm.tetrahedra(rh, plan),
                              rounded_full_H_stereo=mm.stereo_match(rh, Chem.AddHs(Chem.Mol(m))),
                              strict_graph=mm.signature(rt) == mm.signature(m),
                              strict_coordinate_stereo=mm.stereo_match(rt, m),
                              rmsd_A=float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))),
                              max_movement_A=float(np.linalg.norm(delta, axis=1).max()))
                report['eligible'] = bool(eligible(report))
                attempts.append(report)
                candidates[label] = (data, full)
    save(out / 'stage1.json', attempts)
    good = [a for a in attempts if a['eligible']]
    if not good:
        raise ValueError('No stereo-valid ligand minimum; no raw fallback')
    win = min(good, key=lambda a: (a['rmsd_A'], a['arm']))
    data, full = candidates[win['arm']]
    (out / 'prepared_only.pdb').write_bytes(data)
    (out / 'stage1.pdb').unlink()
    (out / 'source_optimized_with_H.rdkit').write_bytes(full.ToBinary(
        Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.CoordsAsDouble))
    save(out / 'stage1_selection.json', {'arm': win['arm'], 'fixed_indices': anchors})
    return win['arm']


def main():
    if sys.flags.optimize:
        raise RuntimeError('Python -O disables verification assertions')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path, help='Input JSON 2.0; paths relative to this file')
    parser.add_argument('output', type=Path)
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument('--md-python', default=None, help='Python with OpenMM, RDKit, NumPy and SciPy')
    runtime.add_argument('--md-image', help='Existing Docker image; never pulled')
    args = parser.parse_args()
    entry, out = args.config.resolve(), args.output.resolve()
    config = validate(entry)
    if any(out == p or out in p.parents for p in [entry] + [Path(r['path']) for r in references(config)]):
        raise ValueError('Output must not contain inputs')
    out.mkdir(parents=True, exist_ok=False)
    try:
        report = execute(out, config, args)
    except Exception as exc:
        report = dict(schema_version=contracts.SCHEMA_VERSION, protocol='0.1',
                      validation_pass=False, stage1_arm=None,
                      error=f'{type(exc).__name__}: {exc}')
    contracts.validate(report, 'report')
    save(out / 'report.json', report)
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report['validation_pass']:
        raise SystemExit(1)


def execute(out, config, args):
    # Snapshot every dependency into a private staging directory. The worker sees
    # only this normalized JSON and its copies, never the original parent layout.
    import shutil
    staged = out / 'inputs'
    staged.mkdir()
    config = json.loads(json.dumps(config))
    for i, ref in enumerate(references(config)):
        target = staged / str(i)
        shutil.copyfile(ref['path'], target)
        ref['path'] = str(i)
    save(staged / 'entry.json', config)
    config = validate(staged / 'entry.json')
    for name, ref in [('system.xml', config['forcefield']['system']),
                      ('atoms.json', config['receptor']['atom_metadata'])]:
        (out / name).write_bytes(gzip.decompress(Path(ref['path']).read_bytes()))
    repo = Path(__file__).resolve().parent
    if args.md_image:
        subprocess.run(['docker', 'image', 'inspect', args.md_image], check=True, stdout=subprocess.DEVNULL)
        cmd = ['docker', 'run', '--rm', '--pull=never', '--network=none', '--cpus=1', '--memory=4g',
               '--user', f'{os.getuid()}:{os.getgid()}', '-e', 'PYTHONDONTWRITEBYTECODE=1',
               '-e', 'OPENMM_CPU_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OMP_NUM_THREADS=1',
               '-v', f'{repo}:/code:ro', '-v', f'{staged}:/inputs:ro', '-v', f'{out}:/output',
               '-w', '/code', '--entrypoint', 'python', args.md_image,
               '-m', 'minimizer.refine', '/inputs/entry.json', '/output']
    else:
        cmd = [args.md_python or sys.executable, '-m', 'minimizer.refine', str(staged / 'entry.json'), str(out)]
    subprocess.run(cmd + ['--validate-only'], cwd=repo, check=True)
    arm = stage1(out, config)
    subprocess.run(cmd, cwd=repo, check=True)
    report = contracts.load_json(out / 'report.json', 'report')
    report['stage1_arm'] = arm
    return contracts.validate(report, 'report')


if __name__ == '__main__':
    main()
