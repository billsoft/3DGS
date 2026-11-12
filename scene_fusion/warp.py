"""Photo-track driven radial depth correction with full covariance transport.

Corrections use only training tracks and preserve source camera rays. The smooth
field has finite support in practice through Gaussian weights and a zero prior.
No room landmarks or object-specific geometry is supplied.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from core import load,write_json

def radial_map(p,center,control,value,h,tree=None):
    tree=tree or cKDTree(control)
    d,ix=tree.query(p,k=min(24,len(control)),workers=4)
    v=np.clip(value[ix],-.18,.18);delta=p[:,None]-control[ix];w=np.exp(-.5*(d/h)**2)
    total=w.sum(1)+2.;field=(w*v).sum(1)/total
    gradient=(-w[:,:,None]*delta/h**2*(v-field[:,None])[:,:,None]).sum(1)/total[:,None]
    s=np.exp(field);ray=p-center
    J=s[:,None,None]*(np.eye(3)[None]+ray[:,:,None]*gradient[:,None,:])
    return center+s[:,None]*ray,J,field

def run(out):
    scene=load(out);C=np.load(out/'aligned-cameras.npz')['C'];folder=out/'warped';folder.mkdir(exist_ok=True);reports=[]
    for i,item in enumerate(scene['stations']):
        a=np.load(out/'aligned'/(item['id']+'.npy'));anchors=np.load(out/item['id']/'depth-anchors.npz');control=anchors['points'];value=anchors['log_ratio']
        if len(control)<12:
            np.save(folder/(item['id']+'.npy'),a)
            reports.append(dict(id=item['id'],anchors=len(control),skipped='Insufficient anchors; retained rigid alignment'))
            continue
        h=np.median(np.linalg.norm(control-C[i],axis=1))*.16;tree=cKDTree(control);b=a.copy();maxmove=0;active=0
        for start in range(0,len(a),40000):
            ids=np.arange(start,min(start+40000,len(a)));p=a[ids,:3]
            mapped,J,field=radial_map(p,C[i],control,value,h,tree)
            determinant=np.linalg.det(J);safe=(determinant>.4)&(determinant<2.5)
            J[~safe]=np.eye(3);mapped[~safe]=p[~safe]
            b[ids,:3]=mapped
            frame=Rotation.from_quat(a[ids][:,[11,12,13,10]]).as_matrix();cov=(frame*np.exp(2*a[ids,None,7:10]))@frame.transpose(0,2,1)
            transformed=J@cov@J.transpose(0,2,1);eig,vec=np.linalg.eigh(transformed)
            vec[:,:,2]*=np.where(np.linalg.det(vec)<0,-1,1)[:,None]
            q=Rotation.from_matrix(vec).as_quat();b[ids,7:10]=.5*np.log(np.maximum(eig,1e-12));b[ids,10:14]=q[:,[3,0,1,2]]
            maxmove=max(maxmove,float(np.linalg.norm(b[ids,:3]-p,axis=1).max()));active+=int((abs(field)>.002).sum())
        assert np.isfinite(b).all();np.save(folder/(item['id']+'.npy'),b)
        reports.append(dict(id=item['id'],anchors=len(control),bandwidth=h,max_shift=maxmove,active=active,
            source_ray_preserved=True,covariance_transport='J covariance J-transpose'))
    write_json(out/'warp-report.json',reports);print(json.dumps(reports),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
