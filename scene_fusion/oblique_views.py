"""Reproducible grazing-angle views and moving routes, independent of camera count."""
import argparse,json
from pathlib import Path
import numpy as np


def make_views(centers,rotations,edges):
    C=np.asarray(centers,dtype=float);Q=np.asarray(rotations,dtype=float)
    if C.ndim!=2 or C.shape[1]!=3 or not len(C) or Q.shape!=(len(C),3,3) or not np.isfinite(C).all() or not np.isfinite(Q).all():raise ValueError('Invalid calibrated camera arrays')
    up=Q[:,:,2].mean(0);up/=np.linalg.norm(up)
    result=[]
    def horizontal(d,i):
        d=d-up*np.dot(d,up)
        if np.linalg.norm(d)<1e-8:
            d=Q[i,:,0]-up*np.dot(Q[i,:,0],up)
        if np.linalg.norm(d)<1e-8:
            d=np.eye(3)[np.argmin(abs(up))];d=d-up*np.dot(d,up)
        return d/np.linalg.norm(d)
    def add(name,c,d,kind):
        z=d/np.linalg.norm(d);x=np.cross(z,up);x/=np.linalg.norm(x)
        basis=np.stack([x,np.cross(z,x),z],axis=1)
        v=np.eye(4);v[:3,:3]=basis.T;v[:3,3]=-basis.T@c
        result.append(dict(name=name,center=c.tolist(),basis=basis.tolist(),viewmat=v.tolist(),kind=kind,source=None))
    for i,c in enumerate(C):
        dist=np.linalg.norm(C-c,axis=1);dist[i]=np.inf;j=int(dist.argmin())
        spacing=dist[j] if len(C)>1 else 1.
        forward=C[j]-c if len(C)>1 else Q[i,:,0]
        forward=horizontal(forward,i)
        side=np.cross(forward,up)
        for sign in [-1,1]:
            origin=c+sign*.12*spacing*side
            for slope,label in [(-.35,'floor'),(.35,'ceiling'),(0.,'wall')]:
                add(f'oblique-{i+1}-{sign}-{label}',origin,forward+slope*up,'oblique')
    for a,b in edges:
        forward=horizontal(C[b]-C[a],a)
        for t in [.2,.4,.5,.6,.8]:
            for slope,label in [(-.4,'floor'),(.4,'ceiling')]:
                add(f'route-{a+1}-{b+1}-{t:.1f}-{label}',(1-t)*C[a]+t*C[b],forward+slope*up,'route')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    z=np.load(a.source/'aligned-cameras.npz');meta=json.loads((a.source/'data-512/dataset.json').read_text('utf8'))
    views=make_views(z['C'],z['Q'],meta['midpoint_edges']);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(views=views),indent=2),encoding='utf8');print('Inspection views',len(views))
