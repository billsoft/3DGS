"""Consensus normals may correct isolated divergent splats, not just agree with them."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
a=np.load(args.input);b=a.copy();frame=Rotation.from_quat(a[:,[11,12,13,10]]).as_matrix().astype('f')
n=frame[np.arange(len(a)),:,a[:,7:10].argmin(1)];tree=cKDTree(a[:,:3]);count=0
for start in range(0,len(a),40000):
    ids=np.arange(start,min(start+40000,len(a)));d,ix=tree.query(a[ids,:3],k=49,workers=4);d=d[:,1:];ix=ix[:,1:]
    scale=np.maximum(np.median(d,axis=1),.0001);w=np.exp(-(d/scale[:,None])**2)*(d<.05)
    cov=np.einsum('bk,bki,bkj->bij',w,n[ix],n[ix]);value,vectors=np.linalg.eigh(cov)
    avg=vectors[:,:,-1];avg*=np.where((avg*n[ids]).sum(1)<0,-1,1)[:,None]
    delta=a[ix,:3]-a[ids,None,:3];off=np.einsum('bkj,bj->bk',delta,avg)
    # Require a close, broad planar neighborhood; two parallel sheets do not vote together.
    w*=np.exp(-(off/np.minimum(.008,scale[:,None]*.5))**2)
    tangent=delta-off[:,:,None]*avg[:,None]
    eig=np.linalg.eigvalsh(np.einsum('bk,bki,bkj->bij',w,tangent,tangent))
    angle=np.arccos(np.clip((avg*n[ids]).sum(1),0,1))
    trusted=(value[:,-1]>.97*value.sum(1))&(eig[:,1]>.15*eig[:,2])&(w.sum(1)>8)&(angle>np.radians(12))
    chosen=ids[trusted];axis=np.cross(n[chosen],avg[trusted]);axis/=np.maximum(np.linalg.norm(axis,axis=1)[:,None],1e-9)
    turn=np.minimum(angle[trusted]*.7,np.radians(25))
    q=Rotation.from_matrix(Rotation.from_rotvec(axis*turn[:,None]).as_matrix()@frame[chosen]).as_quat()
    b[chosen,10:14]=q[:,[3,0,1,2]];count+=len(chosen)
    if start%600000==0:print('Consensus',start,'/',len(a),flush=True)
assert np.array_equal(a[:,:10],b[:,:10]);np.save(args.out/'optimized.npy',b)
(args.out/'consensus.json').write_text(json.dumps(dict(points=len(a),rotated=count,maximum_degrees=25,unchanged_centers_colors_opacity_scales=True),indent=2))
print('Corrected',count,flush=True)
