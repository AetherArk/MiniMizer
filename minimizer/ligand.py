"""Historical PDB binding and anchored MMFF optimizer (adapted imports)."""
import numpy as np
from rdkit import Chem
from scipy.optimize import minimize
from . import stereo as mm

def pdb_binding(path,mapping,smiles):
    raw=path.read_bytes(); lines=raw.decode('ascii').splitlines(keepends=True)
    m=Chem.MolFromSmiles(smiles)
    if m is None: raise ValueError('invalid_source_SMILES')
    graph=mapping['package_graphs']['L']
    if graph['exact_source_smiles']!=smiles: raise ValueError('mapping_source_state_mismatch')
    authority=sorted(graph['atoms'],key=lambda a:a['source_graph_atom_index'])
    if [a['source_graph_atom_index'] for a in authority]!=list(range(m.GetNumAtoms())): raise ValueError('incomplete_mapping')
    atoms={}; edges=set()
    for j,line in enumerate(lines):
        if line.startswith(('ATOM  ','HETATM')):
            if line[16:17].strip(): raise ValueError('unsupported_altloc')
            serial=int(line[6:11])
            if serial in atoms: raise ValueError('duplicate_serial_or_multiple_models')
            atoms[serial]=(j,line)
        elif line.startswith('CONECT'):
            ns=[int(line[k:k+5]) for k in range(6,len(line.rstrip()),5) if line[k:k+5].strip()]
            edges.update(tuple(sorted((ns[0],b))) for b in ns[1:])
    coords=[]; target=[]
    for i,a in enumerate(authority):
        atom = m.GetAtomWithIdx(i)
        state = dict(element=atom.GetSymbol(), formal_charge=atom.GetFormalCharge(),
                     explicit_H=atom.GetNumExplicitHs(), implicit_H=atom.GetNumImplicitHs())
        if any(a[key] != value for key, value in state.items()): raise ValueError('mapping_atom_state_mismatch')
        j,line=atoms[a['package_serial']]
        if line[21:22]!='L' or line[17:20]!='LIG': raise ValueError('pdb_mapping_requires_target_L_LIG')
        if line[12:16].strip()!=a['package_atom_name'] or line[76:78].strip().upper()!=m.GetAtomWithIdx(i).GetSymbol().upper(): raise ValueError('atom_identity_mismatch')
        charge=line[78:80].strip()
        if charge:
            if len(charge)!=2 or charge[0] not in '123456789' or charge[1] not in '+-': raise ValueError('unsupported_PDB_charge_encoding')
            if int(charge[0])*(1 if charge[1]=='+' else -1)!=m.GetAtomWithIdx(i).GetFormalCharge(): raise ValueError('encoded_PDB_charge_mismatch')
        m.GetAtomWithIdx(i).SetProp('source_atom_name',a['package_atom_name'])
        m.GetAtomWithIdx(i).SetProp('molFileAlias',a['package_atom_name'])
        coords.append([float(line[k:k+8]) for k in (30,38,46)]); target.append(j)
    if not np.isfinite(coords).all(): raise ValueError('nonfinite_ligand_coordinates')
    serials={a['package_serial']:i for i,a in enumerate(authority)}
    if {s for s, (_, line) in atoms.items() if line[21:22]=='L'} != set(serials):
        raise ValueError('unmapped_ligand_atoms')
    if any((a in serials)!=(b in serials) for a,b in edges): raise ValueError('covalently_attached_ligand_unsupported')
    observed={tuple(sorted((serials[a],serials[b]))) for a,b in edges if a in serials and b in serials}
    if observed!={tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))) for b in m.GetBonds()}: raise ValueError('explicit_graph_mismatch')
    m=mm.atcoords(m,np.array(coords))
    def writer(x):
        out=list(lines)
        for j,p in zip(target,x):
            xyz=''.join(f'{v:8.3f}' for v in p)
            if len(xyz)!=24: raise ValueError('PDB_coordinate_overflow')
            out[j]=lines[j][:30]+xyz+lines[j][54:]
        return ''.join(out).encode('ascii'),np.round(x,3)
    return m,writer,set(target)


def anchored_optimize(m,h,plan,k,ks,config=mm.DEFAULT):
 fixed=ACTIVE_ANCHORS
 if not fixed:return ORIGINAL_OPT(m,h,plan,k,ks,config)
 x0,fun=mm.objective(m,h,plan,k,ks);moving=np.array([i for i in range(len(x0)) if i not in fixed]);base=x0.copy()
 def f(v):
  x=base.copy();x[moving]=v.reshape(-1,3);e,g=fun(x.ravel());return e,g.reshape(-1,3)[moving].ravel()
 res=minimize(f,x0[moving].ravel(),jac=True,method='L-BFGS-B',options={key:config[key] for key in ('maxiter','maxls','maxcor','ftol','gtol')});x=base.copy();x[moving]=res.x.reshape(-1,3)
 full=mm.atcoords(h,x);moved=mm.atcoords(m,x[:m.GetNumAtoms()]);assert np.array_equal(x[fixed],x0[fixed])
 return moved,full,dict(converged=bool(res.success),message=str(res.message),iterations=int(res.nit),objective=float(res.fun),gradient_max_abs=float(np.max(np.abs(res.jac))),tetrahedra=mm.tetrahedra(full,plan),full_H_stereo=mm.stereo_match(full,Chem.AddHs(m)),heavy_stereo=mm.stereo_match(moved,m),fixed_source_indices=fixed,fixed_exact=True)
ORIGINAL_OPT=mm.optimize;ACTIVE_ANCHORS=[]
