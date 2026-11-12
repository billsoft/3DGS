"""Experimental, image-supported planar resampling with one appearance owner.

No object labels or scene coordinates. The input constraints must carry independent
view verification. A static DC-colored PLY cannot reproduce moving reflections;
selecting one photo deliberately preserves one coherent appearance instead.
"""
import argparse,json
from pathlib import Path
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation
from core import load,write_json
from apply_planes import plane_projection
from surface_export import sample_photo

def plane_samples(corners,normal,spacing):
    u=corners[1]-corners[0];u/=np.linalg.norm(u)
    v=np.cross(normal,u);v/=np.linalg.norm(v)
    polygon=(corners-corners[0])@np.stack([u,v],axis=1)
    x=np.arange(polygon[:,0].min()+spacing/2,polygon[:,0].max(),spacing)
    y=np.arange(polygon[:,1].min()+spacing/2,polygon[:,1].max(),spacing)
    xx,yy=np.meshgrid(x,y);xy=np.c_[xx.ravel(),yy.ravel()]
    edge=np.roll(polygon,-1,axis=0)-polygon
    delta=xy[:,None]-polygon[None]
    cross=edge[None,:,0]*delta[:,:,1]-edge[None,:,1]*delta[:,:,0]
    inside=(cross>=0).all(1)|(cross<=0).all(1)
    return corners[0]+xy[inside,0,None]*u+xy[inside,1,None]*v,np.stack([u,v,normal],axis=1)

def run(source,model,planes_path,out,resolution=480,allow_provisional=False):
    out.mkdir(parents=True,exist_ok=True)
    meta=load(source);c=np.load(source/'aligned-cameras.npz');C,Q=c['C'],c['Q']
    data=json.loads(planes_path.read_text(encoding='utf8'));planes=data['planes']
    if not planes:raise ValueError('No verified bounded surfaces; leave input model unchanged')
    if not 32<=resolution<=2048:raise ValueError('Surface resolution must be in [32, 2048]')
    provisional=data['held_mean_limit']>3.5 or data['held_max_limit']>6
    if provisional and not allow_provisional:raise ValueError('Provisional geometry requires explicit --allow-provisional; not a production result')
    for plane in planes:
        if len(set(plane['cameras']))<3:raise ValueError('A bounded surface requires three distinct supporting cameras')
        if plane['held_mean_px']>data['held_mean_limit'] or plane['held_max_px']>data['held_max_limit']:raise ValueError('Constraint exceeds its declared verification limits')
    a=np.load(model);owner=np.load(source/'source.npy')
    if len(owner)!=len(a):raise ValueError('Source identities do not correspond to model')
    if a.ndim!=2 or a.shape[1]!=14 or not np.isfinite(a).all():raise ValueError('Invalid Gaussian array')
    if owner.ndim!=1 or np.any(owner<0) or np.any(owner>=len(C)):raise ValueError('Invalid source identities')
    keep=np.ones(len(a),bool);patches=[];patchowners=[];report=[]
    for pi,plane in enumerate(planes):
        corners=np.array(plane['corners']);normal=np.array(plane['normal']);normal/=np.linalg.norm(normal)
        center=corners.mean(0);spacing=min(np.linalg.norm(corners[1]-corners[0]),np.linalg.norm(corners[3]-corners[0]))/resolution
        p,frame=plane_samples(corners,normal,spacing)
        scores=[]
        for camera in plane['cameras']:
            d=center-C[camera];distance=np.linalg.norm(d);incidence=abs(d@normal)/distance
            # Photo angular resolution and incidence; never a semantic preference.
            scores.append(incidence/max(distance,.1)**2)
        best=plane['cameras'][int(np.argmax(scores))]
        image=cv.imdecode(np.fromfile(meta['stations'][best]['photo'],np.uint8),1)
        rgb=sample_photo(image,p,C[best],Q[best])
        patch=np.empty((len(p),14),'f');patch[:,:3]=p;patch[:,3:6]=(rgb-.5)/.2820947918
        patch[:,6]=6.;patch[:,7:10]=np.log([spacing*.75,spacing*.75,spacing*.03])
        patch[:,10:14]=Rotation.from_matrix(frame).as_quat()[[3,0,1,2]]
        removed=0
        for camera in range(len(C)):
            ids=np.flatnonzero((owner==camera)&keep)
            for start in range(0,len(ids),30000):
                ix=ids[start:start+30000];_,_,inside,_=plane_projection(a[ix,:3],C[camera],corners,normal)
                keep[ix[inside]]=False;removed+=int(inside.sum())
        patches.append(patch);patchowners.append(np.full(len(patch),best,'i2'))
        report.append(dict(plane=pi,owner=best,owner_scores=scores,removed=removed,inserted=len(p),spacing=spacing,
                           held_mean_px=plane['held_mean_px'],held_max_px=plane['held_max_px']))
    result=np.concatenate([a[keep]]+patches);np.save(out/'optimized.npy',result)
    np.save(out/'source.npy',np.concatenate([owner[keep]]+patchowners))
    np.save(out/'geometry-protected.npy',np.r_[np.zeros(keep.sum(),bool),np.ones(sum(map(len,patches)),bool)])
    write_json(out/'resampling.json',dict(input_points=len(a),points=len(result),planes=report,
        constraints=str(planes_path),held_mean_limit=data['held_mean_limit'],held_max_limit=data['held_max_limit'],provisional=provisional,
        semantic_rules=False,manual_coordinates=False,appearance='One image per bounded surface; fixed DC colors; reflections do not vary with view'))
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--planes',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--resolution',type=int,default=480);p.add_argument('--allow-provisional',action='store_true')
    a=p.parse_args();run(a.source,a.model,a.planes,a.out,a.resolution,a.allow_provisional)
