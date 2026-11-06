"""Historical local objective, with explicit packet directory."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import openmm as mm
from openmm import unit as u

def build(d):
 d=Path(d);full=mm.XmlSerializer.deserialize((d/'system.xml').read_text());x0=np.load(d/'starting_positions.npy');meta=json.loads((d/'atoms.json').read_text());mob=json.loads((d/'mobility.json').read_text());n=mob['ligand_offset'];nh=mob['ligand_heavy'];lp=x0[n:n+nh]
 selected={a['resid'] for a in meta if a['chain']=='A' and a['element']!='H' and np.min(np.linalg.norm(lp-x0[a['index']],axis=1))<=.5}
 pocket={a['index'] for a in meta if a['chain']=='H' or (a['chain']=='A' and (a['resid'] in selected or a['resid']=='442'))}|set(range(n,len(x0)))
 # Reuse qualified backbone/axial restrictions but restrict strictly to selected residues.
 moving={i for i in mob['movable'] if i>=n or meta[i]['resid'] in selected};retained=set(pocket);bonded=[];skip=(mm.NonbondedForce,mm.CustomNonbondedForce,mm.CustomExternalForce)
 for f in full.getForces():
  if isinstance(f,skip):continue
  root=ET.fromstring(mm.XmlSerializer.serialize(f));found=False
  for parent in root.iter():
   for el in list(parent):
    indices=[int(v) for k,v in el.attrib.items() if k in ('p1','p2','p3','p4','p5','p6','p7','p8')]
    if not indices:continue
    found=True
    if not moving.intersection(indices):parent.remove(el)
    else:retained.update(indices)
  if found:bonded.append(root)
 ids=sorted(retained);remap={v:i for i,v in enumerate(ids)};s=mm.System()
 for i in ids:s.addParticle(full.getParticleMass(i) if i in moving else 0)
 for root in bonded:
  for el in root.iter():
   for k in list(el.attrib):
    if k in ('p1','p2','p3','p4','p5','p6','p7','p8'):el.set(k,str(remap[int(el.get(k))]))
  s.addForce(mm.XmlSerializer.deserialize(ET.tostring(root,encoding='unicode')))
 # Native receptor Coulomb exceptions and CHARMM NBFIX coefficients.
 nb=next(f for f in full.getForces() if isinstance(f,mm.NonbondedForce));lj=next(f for f in full.getForces() if isinstance(f,mm.CustomNonbondedForce));cross=[f for f in full.getForces() if isinstance(f,mm.CustomNonbondedForce)][1]
 size,_,av=lj.getTabulatedFunction(0).getFunctionParameters();_,_,bv=lj.getTabulatedFunction(1).getFunctionParameters();ex={tuple(sorted((int(a),int(b)))):q.value_in_unit(u.elementary_charge**2) for a,b,q,si,e in [nb.getExceptionParameters(i) for i in range(nb.getNumExceptions())]};lex={tuple(sorted(lj.getExclusionParticles(i))) for i in range(lj.getNumExclusions())};p=np.array([cross.getParticleParameters(i) for i in range(len(x0))]);types=[int(lj.getParticleParameters(i)[0]) for i in range(n)]
 pair=mm.CustomBondForce('138.93545764438198*q/r+a/r^12-b/r^6')
 for k in ('q','a','b'):pair.addPerBondParameter(k)
 npairs=0
 for i in sorted(pocket):
  for j in sorted(pocket):
   if j>=i:break
   if not ({i,j}&moving) or (i>=n and j>=n):continue # ligand internal already copied exactly
   if i<n:
    q=ex.get((j,i),p[i,0]*p[j,0]);a=av[types[i]+types[j]*size];b=bv[types[i]+types[j]*size]
    if (j,i) in lex:a=b=0.
   else:
    q=p[i,0]*p[j,0];sig=(p[i,1]+p[j,1])/2;eps=np.sqrt(p[i,2]*p[j,2]);a=4*eps*sig**12;b=4*eps*sig**6
   if q or a or b:pair.addBond(remap[i],remap[j],[q,a,b]);npairs+=1
 s.addForce(pair)
 restr=mm.CustomExternalForce('0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)');restr.addGlobalParameter('k',12552.)
 for k in ('x0','y0','z0'):restr.addPerParticleParameter(k)
 for i in sorted(moving):
  if i<n and meta[i]['element']=='H' or i>=n+nh:continue
  restr.addParticle(remap[i],x0[i])
 s.addForce(restr)
 contract=dict(selected_residues=sorted(selected,key=int),selection='whole protein residues with any initial heavy atom <=5.0 A from initial ligand heavy atoms; frozen',global_indices=ids,moving_global=sorted(moving),moving_local=[remap[i] for i in sorted(moving)],boundary_global=sorted(retained-pocket),fixed_axial_environment_residues=['442'] if '442' not in selected else [],fixed_heme_global=[a['index'] for a in meta if a['chain']=='H'],nonbonded_pair_count=npairs,protein_residue_distances_A={r:min(float(np.min(np.linalg.norm(lp-x0[a['index']],axis=1))*10) for a in meta if a['chain']=='A' and a['resid']==r and a['element']!='H') for r in sorted(selected,key=int)},boundary_protocol='No topology retyping or caps. Retain original full-FF bonded terms with any moving atom, with all endpoints; outside endpoints fixed. Original charges, exclusions, 1-4, NBFIX retained. Nonbonded pairs only among selected whole residues + fixed heme + fixed axial Cys442 when outside selection + ligand with >=1 moving endpoint. Fixed-fixed terms omitted constants; distant protein nonbonded interactions deliberately excluded.',ligand_offset=n,ligand_heavy=nh)
 return s,x0,contract
