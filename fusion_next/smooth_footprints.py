"""Damp isolated Gaussian normal/extent outliers without moving or recoloring.

Only coherent local tangent neighborhoods may vote. This is a proposal generator;
coverage and source-photo checks decide whether its output can be published.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def run():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    a=np.load(args.input);b=a.copy();c=a.copy();pos=a[:,:3]
    frames=Rotation.from_quat(a[:,[11,12,13,10]]).as_matrix().astype('f')
    n=frames[np.arange(len(a)),:,a[:,7:10].argmin(1)]
    tree=cKDTree(pos);rotated=0;shortened=0
    for start in range(0,len(a),60000):
        ids=np.arange(start,min(start+60000,len(a)));dist,ix=tree.query(pos[ids],k=25,workers=4);dist=dist[:,1:];ix=ix[:,1:]
        delta=pos[ix]-pos[ids,None];dot=np.einsum('bkj,bj->bk',n[ix],n[ids]);off=np.einsum('bkj,bj->bk',delta,n[ids])
        spacing=np.maximum(np.median(dist,axis=1),.0001)
        valid=(abs(dot)>.8)&(dist<.035)&(abs(off)<np.minimum(.006,spacing[:,None]*.5))
        w=valid*np.exp(-(dist/(spacing[:,None]+1e-9))**2)
        avg=(w[:,:,None]*np.sign(dot)[:,:,None]*n[ix]).sum(1)/(w.sum(1)[:,None]+1e-9)
        coherence=np.linalg.norm(avg,axis=1);avg/=np.maximum(coherence[:,None],1e-9)
        angle=np.arccos(np.clip((avg*n[ids]).sum(1),-1,1))
        tangent=delta-off[:,:,None]*n[ids,None]
        eig=np.linalg.eigvalsh(np.einsum('bk,bki,bkj->bij',w,tangent,tangent))
        balance=np.linalg.norm((w[:,:,None]*tangent).sum(1)/(w.sum(1)[:,None]+1e-9),axis=1)
        support=(valid.sum(1)>=8)&(eig[:,1]>.15*eig[:,2])&(balance<spacing*.3)&(coherence>.985)
        take=support&(angle>np.radians(6))&(angle<np.radians(30))
        axis=np.cross(n[ids[take]],avg[take]);axis/=np.maximum(np.linalg.norm(axis,axis=1)[:,None],1e-9)
        rotation=Rotation.from_rotvec(axis*np.minimum(angle[take]*.5,np.radians(5))[:,None])
        q=Rotation.from_matrix(rotation.as_matrix()@frames[ids[take]]).as_quat()
        b[ids[take],10:14]=q[:,[3,0,1,2]];rotated+=int(take.sum())
        median=np.median(a[ix,7:10].max(2),axis=1)
        largest=a[ids,7:10].argmax(1);excess=a[ids,7:10].max(1)-median-np.log(1.8)
        tail=support&(excess>0)
        c[ids[tail],7+largest[tail]]-=np.minimum(excess[tail]*.5,np.log(1.15));shortened+=int(tail.sum())
        if start%600000==0:print('Footprints',start,'/',len(a),flush=True)
    combined=c.copy();combined[:,10:14]=b[:,10:14]
    for name,data in [('orientation',b),('tail',c),('combined',combined)]:
        assert np.array_equal(data[:,:7],a[:,:7]);assert np.isfinite(data).all();np.save(args.out/f'{name}.npy',data)
    report=dict(points=len(a),rotated=rotated,shortened=shortened,max_rotation_degrees=5,
        max_extent_reduction=1-1/1.15,centers_colors_opacity_exactly_unchanged=True)
    (args.out/'footprints.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':run()
