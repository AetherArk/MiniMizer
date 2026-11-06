"""Stage 2: prepared force fields, fixed receptor states, fresh restraint references."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENMM_CPU_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
from pathlib import Path
import sys
import numpy as np
from rdkit import Chem
import openmm as mm
from openmm import unit as u
from scipy.optimize import minimize
from .pocket import build
from .boundary import verify


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(packet, out):
    config = json.loads((packet / 'config.json').read_text())
    meta = json.loads((out / 'atoms.json').read_text())
    mapping = config['mapping']
    lig = Chem.Mol((out / 'source_optimized_with_H.rdkit').read_bytes())
    lines = (out / 'prepared_only.pdb').read_text().splitlines(keepends=True)
    byserial = {int(l[6:11]): l for l in lines if l.startswith(('ATOM  ', 'HETATM'))}
    for a in mapping['package_graphs']['L']['atoms']:
        line = byserial[a['package_serial']]
        lig.GetConformer().SetAtomPosition(a['source_graph_atom_index'],
                                          [float(line[k:k+8]) for k in (30, 38, 46)])
    # Historical prepare.py writes the default (float32-coordinate) RDKit pickle.
    # md_worker.py reads it before forming x0. Do not silently remove this rounding.
    lig = Chem.Mol(lig.ToBinary())
    from .stereo import signature
    authority = Chem.AddHs(Chem.MolFromSmiles(config['source_smiles']))
    if signature(lig) != signature(authority):
        raise ValueError('Ligand state or atom order changed')
    state = config['ligand_state']
    if (Chem.GetFormalCharge(lig) != state['source_formal_charge']
            or lig.GetNumAtoms() != state['atom_count']
            or sum(a.GetAtomicNum() == 1 for a in lig.GetAtoms()) != state['H_count']):
        raise ValueError('Ligand force-field state mismatch')
    (out / 'ligand.rdkit').write_bytes(lig.ToBinary())
    receptor = np.load(packet / 'receptor-positions.npy', allow_pickle=False)
    n, nh = len(receptor), lig.GetNumHeavyAtoms()
    x0 = np.concatenate([receptor, lig.GetConformer().GetPositions() / 10])
    lp = x0[n:n+nh]
    selected = {a['resid'] for a in meta if a['chain'] == 'A' and a['element'] != 'H'
                and np.min(np.linalg.norm(lp-x0[a['index']], axis=1)) <= .5}
    backbone = {'N', 'CA', 'C', 'O', 'OXT', 'H', 'H1', 'H2', 'H3', 'HA', 'HA2', 'HA3'}
    anchors = config['fixed_ligand_source_indices']
    movable = [a['index'] for a in meta if a['chain'] == 'A' and a['resid'] in selected
               and a['name'] not in backbone and not (a['resid'] == '442' and a['name'] == 'SG')]
    movable += [n+i for i in range(lig.GetNumAtoms()) if i not in anchors]
    # Rebuild only coordinate-dependent restraints; preserve all frozen chemistry.
    full = mm.XmlSerializer.deserialize((out / 'system.xml').read_text())
    if full.getNumParticles() != len(x0):
        raise ValueError('Force-field particle count mismatch')
    external = [i for i, f in enumerate(full.getForces()) if isinstance(f, mm.CustomExternalForce)]
    if len(external) != 1:
        raise ValueError('Expected one positional restraint force')
    old = full.getForce(external[0])
    if old.getEnergyFunction() != '0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)':
        raise ValueError('Unsupported external force')
    full.removeForce(external[0])
    restraint = mm.CustomExternalForce('0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.addGlobalParameter('k', 12552.)
    for name in ('x0', 'y0', 'z0'):
        restraint.addPerParticleParameter(name)
    for i in sorted(movable):
        if (i < n and meta[i]['element'] == 'H') or i >= n+nh:
            continue
        restraint.addParticle(i, x0[i])
    full.addForce(restraint)
    (out / 'system.xml').write_text(mm.XmlSerializer.serialize(full))
    np.save(out / 'starting_positions.npy', x0)
    save(out / 'mobility.json', dict(movable=movable, ligand_offset=n, ligand_heavy=nh))
    s, x0, contract = build(out)
    ids = np.array(contract['global_indices'])
    mov = np.array(contract['moving_local'])
    globalmov = np.array(contract['moving_global'])
    xs = x0[ids]
    local = out / 'local'
    local.mkdir()
    save(local / 'pocket_contract.json', contract)
    (local / 'system.xml').write_text(mm.XmlSerializer.serialize(s))
    context = mm.Context(s, mm.VerletIntegrator(.001), mm.Platform.getPlatformByName('Reference'))
    context.setParameter('k', 12552.)
    calls = 0

    def fun(flat):
        nonlocal calls
        xx = xs.copy()
        xx[mov] = flat.reshape(-1, 3)
        context.setPositions(xx*u.nanometer)
        state = context.getState(getEnergy=True, getForces=True)
        calls += 1
        return (state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole),
                -state.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole/u.nanometer)[mov].ravel())

    ei, _ = fun(xs[mov].ravel())
    res = minimize(fun, xs[mov].ravel(), jac=True, method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 10., 'maxls': 30, 'maxcor': 10})
    ef, gf = fun(res.x)
    x = x0.copy()
    x[globalmov] = res.x.reshape(-1, 3)
    fixed = np.array(sorted(set(range(len(x)))-set(globalmov)))
    opt = dict(k_kJ_mol_nm2=12552., initial_energy_kJ_mol=ei, final_energy_kJ_mol=ef,
               scipy_success=bool(res.success), message=str(res.message), iterations=int(res.nit),
               evaluations=calls, max_gradient_kJ_mol_nm=float(np.max(np.abs(gf))),
               rms_gradient_kJ_mol_nm=float(np.sqrt(np.mean(gf**2))),
               fixed_max_movement_nm=float(np.max(np.abs(x[fixed]-x0[fixed]))))
    arm = local / 'k12552'
    arm.mkdir()
    np.save(arm / 'positions.npy', x)
    save(arm / 'optimization.json', opt)
    index = {(a['chain'], a['resid'], a['name']): a['index'] for a in meta}
    si = {a['package_serial']: n+a['source_graph_atom_index'] for a in mapping['package_graphs']['L']['atoms']}
    output = []
    for line in lines:
        if line.startswith(('ATOM  ', 'HETATM')):
            idx = si.get(int(line[6:11])) if line[21] == 'L' else index.get(
                (line[21], line[22:26].strip(), line[12:16].strip())) if line[21] == 'A' else None
            if idx is not None:
                line = line[:30]+''.join(f'{z:8.3f}' for z in x[idx]*10)+line[54:]
        output.append(line)
    candidate = arm / 'candidate.pdb'
    candidate.write_text(''.join(output))
    for i in range(lig.GetNumAtoms()):
        lig.GetConformer().SetAtomPosition(i, x[n+i]*10)
    (arm / 'optimized_with_H.rdkit').write_bytes(lig.ToBinary())
    # Keep the historical 1e-5 kJ/mol boundary gate unchanged. No physical-QC claim.
    boundary_pass, boundary_error = True, None
    try:
        verify(out, local)
    except AssertionError as exc:
        boundary_pass, boundary_error = False, str(exc)
    report = dict(protocol='0.1', optimization=opt,
                  candidate='local/k12552/candidate.pdb',
                  boundary_pass=boundary_pass, boundary_error=boundary_error,
                  boundary_threshold_kJ_mol=1e-5,
                  physical_checks='PoseBusters not run',
                  validation_pass=bool(boundary_pass and opt['scipy_success']
                                       and opt['fixed_max_movement_nm'] == 0.))
    save(out / 'report.json', report)


if __name__ == '__main__':
    run(Path(sys.argv[1]), Path(sys.argv[2]))
