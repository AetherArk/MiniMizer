"""Source-derived, ligand-only stereo-restrained MMFF94s. No embedding/fallback.
Source tags are never modified; equivalence probes operate on separate copies.
"""
from itertools import combinations, product
import numpy as np
from scipy.optimize import minimize
from rdkit import Chem
from rdkit.Chem import AllChem

VERSION = 'source_stereo_mmff94s_four_faces_v1'
DEFAULT = dict(version=VERSION, position_k=[10.,30.,100.,300.], stereo_k=[100.,1000.],
               max_relative_centers=4, rmsd_limit_A=0.5, max_atom_limit_A=1.0,
               maxiter=3000, maxls=50, maxcor=20, ftol=1e-13, gtol=1e-4)

class Unsupported(ValueError): pass

def signature(m):
    return (Chem.MolToSmiles(Chem.Mol(m)), tuple((a.GetAtomicNum(),a.GetFormalCharge(),a.GetIsotope(),
        int(a.GetChiralTag()),a.GetNumExplicitHs(),a.GetNumImplicitHs()) for a in m.GetAtoms()),
        tuple((b.GetBeginAtomIdx(),b.GetEndAtomIdx(),str(b.GetBondType()),int(b.GetStereo()),
               tuple(b.GetStereoAtoms())) for b in m.GetBonds()))

def atcoords(m,x):
    m=Chem.Mol(m); c=Chem.Conformer(len(x)); c.Set3D(True)
    for i,p in enumerate(x): c.SetAtomPosition(i,p)
    m.RemoveAllConformers(); m.AddConformer(c); return m

def perceived(m):
    p=Chem.Mol(m); Chem.RemoveStereochemistry(p)
    Chem.AssignStereochemistryFrom3D(p,replaceExistingTags=True)
    return p

def stereo_match(m,authority):
    # Full chirality-aware graph match catches relative rings without CIP labels.
    p=perceived(m)
    return p.GetNumAtoms()==authority.GetNumAtoms() and p.HasSubstructMatch(authority,useChirality=True)

def plans(m,config=DEFAULT):
    if m.GetStereoGroups(): raise Unsupported('enhanced_stereo_groups')
    if len(Chem.GetMolFrags(m))!=1: raise Unsupported('disconnected_ligand_state')
    if any(a.GetNumRadicalElectrons() or a.GetIsotope() or a.GetAtomicNum()==1 for a in m.GetAtoms()):
        raise Unsupported('radical_isotopic_or_explicit_H_source_state')
    if any(b.GetStereo()!=Chem.BondStereo.STEREONONE for b in m.GetBonds()):
        raise Unsupported('specified_EZ_or_other_bond_stereo_not_yet_constrained')
    centers=[]; relative=[]
    for a in m.GetAtoms():
        tag=a.GetChiralTag()
        if tag==Chem.ChiralType.CHI_UNSPECIFIED: continue
        if tag not in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            raise Unsupported('non_tetrahedral_stereo')
        if a.GetAtomicNum()!=6: raise Unsupported('noncarbon_tetrahedral_center')
        centers.append(a.GetIdx())
        if not a.HasProp('_CIPCode'): relative.append(a.GetIdx())
    if len(relative)>config['max_relative_centers']: raise Unsupported('relative_stereo_enumeration_bound')
    h=Chem.AddHs(Chem.Mol(m),addCoords=True)
    if not AllChem.MMFFHasAllMoleculeParams(h): raise Unsupported('unsupported_MMFF94s_parameters')
    result=[]
    for bits in product((False,True),repeat=len(relative)):
        probe=Chem.Mol(m)
        for i,flip in zip(relative,bits):
            if flip: probe.GetAtomWithIdx(i).InvertChirality()
        # Canonical SMILES may reuse stale _ringStereoAtoms after tag inversion.
        # Chirality-aware full-graph matching is the equivalence authority.
        if not probe.HasSubstructMatch(m,useChirality=True): continue
        faces=[]; sites=[]
        for i in centers:
            ns=[a.GetIdx() for a in h.GetAtomWithIdx(i).GetNeighbors()]
            if len(ns)!=4: raise Unsupported('tetrahedral_center_not_four_neighbors_with_H')
            s=1 if probe.GetAtomWithIdx(i).GetChiralTag()==Chem.ChiralType.CHI_TETRAHEDRAL_CCW else -1
            sites.append((i,ns))
            for j,triple in enumerate(combinations(ns,3)):
                heavy=all(h.GetAtomWithIdx(k).GetAtomicNum()>1 for k in triple)
                faces.append((i,tuple(triple),s*((1,-1,1,-1)[j]),2. if heavy else 1.,heavy))
        result.append(dict(faces=faces,sites=sites,equivalent_relative_inversions=list(bits)))
    if not result: raise Unsupported('no_source_equivalent_orientation')
    return h,result

def volumes_grad(x,faces):
    values=[]; grads=[]
    for center,(ia,ib,ic),sign,margin,heavy in faces:
        a,b,c=x[ia]-x[center],x[ib]-x[center],x[ic]-x[center]
        ga,gb,gc=np.cross(b,c),np.cross(c,a),np.cross(a,b)
        g=np.zeros_like(x); g[ia]+=ga;g[ib]+=gb;g[ic]+=gc;g[center]-=ga+gb+gc
        values.append(np.dot(np.cross(a,b),c));grads.append(g)
    return np.array(values),np.array(grads).reshape((-1,)+x.shape)

def objective(m,h,plan,k,ks):
    h=Chem.Mol(h); n=m.GetNumAtoms(); x0=h.GetConformer().GetPositions().copy()
    assert np.array_equal(x0[:n],m.GetConformer().GetPositions())
    props=AllChem.MMFFGetMoleculeProperties(h,mmffVariant='MMFF94s')
    ff=AllChem.MMFFGetMoleculeForceField(h,props,nonBondedThresh=100.,ignoreInterfragInteractions=True);ff.Initialize()
    signs=np.array([f[2] for f in plan['faces']]); margins=np.array([f[3] for f in plan['faces']])
    def fun(flat):
        x=np.asarray(flat).reshape(-1,3); e=ff.CalcEnergy(flat.tolist()); g=np.array(ff.CalcGrad(flat.tolist())).reshape(-1,3)
        delta=x[:n]-x0[:n];e+=.5*k*np.sum(delta**2);g[:n]+=k*delta
        if len(signs):
            v,vg=volumes_grad(x,plan['faces']);missing=np.maximum(0.,margins-signs*v)
            e+=.5*ks*np.sum(missing**2);g-=ks*np.sum((missing*signs)[:,None,None]*vg,axis=0)
        return float(e),g.ravel()
    return x0,fun

def tetrahedra(h,plan):
    x=h.GetConformer().GetPositions(); out=[]
    for center,ns in plan['sites']:
        try: weights=np.linalg.solve(np.vstack([x[ns].T,np.ones(4)]),np.r_[x[center],1.])
        except np.linalg.LinAlgError: weights=np.full(4,np.nan)
        out.append(dict(center=center,neighbors=ns,weights=weights.tolist(),valid=bool(np.isfinite(weights).all() and np.all(weights>0))))
    volumes,_=volumes_grad(x,plan['faces'])
    oriented=[float(f[2]*v) for f,v in zip(plan['faces'],volumes)]
    valid=all(r['valid'] for r in out) and all(v>0 and (not f[4] or v>=1.5) for f,v in zip(plan['faces'],oriented))
    return dict(valid=valid,sites=out,oriented_volumes_A3=oriented)

def optimize(m,h,plan,k,ks,config=DEFAULT):
    before=signature(m); xyz=m.GetConformer().GetPositions().copy()
    x0,fun=objective(m,h,plan,k,ks)
    res=minimize(fun,x0.ravel(),jac=True,method='L-BFGS-B',options={k:config[k] for k in ('maxiter','maxls','maxcor','ftol','gtol')})
    h=atcoords(h,res.x.reshape(-1,3)); moved=atcoords(m,h.GetConformer().GetPositions()[:len(xyz)])
    assert signature(m)==before and np.array_equal(xyz,m.GetConformer().GetPositions())
    # CXSMILES includes coordinates: compare chemical signature on conformer-free copies downstream.
    return moved,h,dict(converged=bool(res.success),message=str(res.message),iterations=int(res.nit),
        objective=float(res.fun),gradient_max_abs=float(np.max(np.abs(res.jac))),
        tetrahedra=tetrahedra(h,plan),full_H_stereo=stereo_match(h,Chem.AddHs(Chem.Mol(m))),
        heavy_stereo=stereo_match(moved,m))
