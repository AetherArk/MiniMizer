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
from minimizer import ligand


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate(packet):
    config = json.loads((packet / 'config.json').read_text())
    if config['protocol'] != '0.1':
        raise ValueError('Unsupported protocol')
    for name in ('input.pdb', 'system.xml.gz', 'atoms.json.gz', 'receptor-positions.npy'):
        if not (packet / name).is_file():
            raise ValueError('Missing prepared input: ' + name)
    receptor = np.load(packet / 'receptor-positions.npy', allow_pickle=False)
    meta = json.loads(gzip.decompress((packet / 'atoms.json.gz').read_bytes()))
    if receptor.shape != (len(meta), 3) or not np.isfinite(receptor).all():
        raise ValueError('Invalid receptor coordinates')
    if [a['index'] for a in meta] != list(range(len(meta))):
        raise ValueError('Invalid receptor atom ordering')
    identities = {(a['chain'], a['resid'], a['name']): a for a in meta}
    if len(identities) != len(meta):
        raise ValueError('Duplicate receptor identity')
    seen = set()
    for line in (packet / 'input.pdb').read_text().splitlines():
        if not line.startswith(('ATOM  ', 'HETATM')) or line[21] == 'L':
            continue
        name = line[12:16].strip()
        if line[21] == 'H':
            name = config['heme_pdb_names'].get(name, name)
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
    ligand.pdb_binding(packet / 'input.pdb', config['mapping'], config['source_smiles'])
    return config


def eligible(a):
    return (a['converged'] and a['tetrahedra']['valid'] and a['rounded_tetrahedra']['valid']
            and a['full_H_stereo'] and a['rounded_full_H_stereo'] and a['heavy_stereo']
            and a['strict_graph'] and a['strict_coordinate_stereo']
            and a['rmsd_A'] <= .5 and a['max_movement_A'] <= 1.)


def stage1(packet, out, config):
    mapping = config['mapping']
    m, writer, _ = ligand.pdb_binding(packet / 'input.pdb', mapping, config['source_smiles'])
    original = m.GetConformer().GetPositions().copy()
    anchors = config['fixed_ligand_source_indices']
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
                rt, _, _ = ligand.pdb_binding(tmp, mapping, config['source_smiles'])
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
    parser.add_argument('packet', type=Path)
    parser.add_argument('output', type=Path)
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument('--md-python', default=None, help='Python with OpenMM, RDKit, NumPy and SciPy')
    runtime.add_argument('--md-image', help='Existing Docker image; never pulled')
    args = parser.parse_args()
    packet, out = args.packet.resolve(), args.output.resolve()
    config = validate(packet)
    if out == packet or out in packet.parents:
        raise ValueError('Output must not contain packet')
    out.mkdir(parents=True, exist_ok=False)
    arm = stage1(packet, out, config)
    for name in ('system.xml', 'atoms.json'):
        (out / name).write_bytes(gzip.decompress((packet / (name + '.gz')).read_bytes()))
    repo = Path(__file__).resolve().parent
    if args.md_image:
        subprocess.run(['docker', 'image', 'inspect', args.md_image], check=True, stdout=subprocess.DEVNULL)
        cmd = ['docker', 'run', '--rm', '--pull=never', '--network=none', '--cpus=1', '--memory=4g',
               '--user', f'{os.getuid()}:{os.getgid()}', '-e', 'PYTHONDONTWRITEBYTECODE=1',
               '-e', 'OPENMM_CPU_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OMP_NUM_THREADS=1',
               '-v', f'{repo}:/code:ro', '-v', f'{packet}:/packet:ro', '-v', f'{out}:/output',
               '-w', '/code', '--entrypoint', 'python', args.md_image,
               '-m', 'minimizer.refine', '/packet', '/output']
    else:
        cmd = [args.md_python or sys.executable, '-m', 'minimizer.refine', str(packet), str(out)]
    subprocess.run(cmd, cwd=repo, check=True)
    report = json.loads((out / 'report.json').read_text())
    report['stage1_arm'] = arm
    save(out / 'report.json', report)
    print(json.dumps(report, indent=2))
    if not report['validation_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
