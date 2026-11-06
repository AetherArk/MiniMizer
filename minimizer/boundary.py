#!/usr/bin/env python3
from .pocket import *

def save(p, value):
 p.write_text(json.dumps(value, indent=2)+"\n")

def energy(s,x):
 c=mm.Context(s,mm.VerletIntegrator(.001),mm.Platform.getPlatformByName('Reference'));c.setPositions(x*u.nanometer);st=c.getState(getEnergy=True,getForces=True);return st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole),st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole/u.nanometer)
def verify(prepared,d):
 lid=prepared.name;V3=V2=prepared.parent;rows=[]
 s,x0,contract=build(prepared);ids=np.array(contract['global_indices']);full=mm.XmlSerializer.deserialize((V3/lid/'system.xml').read_text());x=np.load(d/'k12552/positions.npy');n=contract['ligand_offset'];moving=set(contract['moving_global']);meta=json.loads((V3/lid/'atoms.json').read_text());outside=[a['index'] for a in meta if a['chain']=='A' and a['resid'] not in contract['selected_residues'] and a['resid']!='442']
 nb=next(f for f in full.getForces() if isinstance(f,mm.NonbondedForce));lj,cross=[f for f in full.getForces() if isinstance(f,mm.CustomNonbondedForce)];size,_,av=lj.getTabulatedFunction(0).getFunctionParameters();_,_,bv=lj.getTabulatedFunction(1).getFunctionParameters();p=np.array([cross.getParticleParameters(i) for i in range(len(x))]);typ=[int(lj.getParticleParameters(i)[0]) for i in range(n)];exc={tuple(sorted((int(a),int(b)))):q.value_in_unit(u.elementary_charge**2) for a,b,q,si,e in [nb.getExceptionParameters(i) for i in range(nb.getNumExceptions())]};lex={tuple(sorted(lj.getExclusionParticles(i))) for i in range(lj.getNumExclusions())}
 def omitted(xx):
  value=0.
  for i in sorted(moving):
   for j in outside:
    dist=np.linalg.norm(xx[i]-xx[j]);key=tuple(sorted((i,j)))
    if i<n:
     q=exc.get(key,p[i,0]*p[j,0]);a=av[typ[i]+typ[j]*size];b=bv[typ[i]+typ[j]*size]
     if key in lex:a=b=0.
    else:
     q=p[i,0]*p[j,0];sigma=(p[i,1]+p[j,1])/2;eps=np.sqrt(p[i,2]*p[j,2]);a=4*eps*sigma**12;b=4*eps*sigma**6
    value+=138.93545764438198*q/dist+a/dist**12-b/dist**6
  return value
 e0,f0=energy(s,x0[ids]);e,f=energy(s,x[ids]);fe0,_=energy(full,x0);fe,_=energy(full,x);removed=omitted(x)-omitted(x0);err=(fe-fe0)-removed-(e-e0);assert abs(err)<1e-5,(lid,err)
 i=contract['moving_local'][0];xp=x[ids].copy();xm=x[ids].copy();xp[i,0]+=1e-5;xm[i,0]-=1e-5;fd=(energy(s,xp)[0]-energy(s,xm)[0])/(2e-5);gerr=abs(fd+f[i,0]);assert gerr<.1
 selected=set(contract['selected_residues']);assert all(np.array_equal(x[a['index']],x0[a['index']]) for a in meta if a['chain']=='A' and a['resid'] not in selected)
 # Independent selection from initial original heavy PDB, not generated H or relaxed coordinates.
 source=(V2/lid/'prepared_only.pdb').read_text().splitlines();la=np.array([[float(l[k:k+8]) for k in (30,38,46)] for l in source if l.startswith(('ATOM  ','HETATM')) and l[21]=='L']);expected={l[22:26].strip() for l in source if l.startswith('ATOM  ') and l[21]=='A' and min(np.linalg.norm(la-np.array([float(l[k:k+8]) for k in (30,38,46)]),axis=1))<=5};assert expected==selected
 moved=sorted({a['resid'] for a in meta if a['chain']=='A' and np.linalg.norm(x[a['index']]-x0[a['index']])>1e-12},key=int);assert set(moved)<=selected
 rows.append(dict(ligand_id=lid,selected_residues=sorted(selected,key=int),moved_residues=moved,independent_5A_selection_exact=True,outside_selected_all_atoms_exact=True,subset_atoms=len(ids),boundary_atoms=contract['boundary_global'],full_delta_energy=fe-fe0,omitted_distant_nonbonded_delta=removed,pocket_delta_energy=e-e0,energy_delta_identity_error=err,finite_difference_gradient_error=gerr))
 save(d/'boundary_and_selection_verification.json',rows);print(json.dumps(rows,indent=2))
