"""Consolidate well-supported planar depth without spanning foreground edges.

Planes are estimated from the scene itself, never floor/ceiling coordinates or
object labels. Only existing high-confidence, locally aligned samples move;
unknown depths are not filled. This is a geometric prior, not measured depth.
"""
import argparse,json,shutil,hashlib
from pathlib import Path
import numpy as np


def backproject(depth,V):
    h,w=depth.shape;y,x=np.mgrid[:h,:w];rays=np.stack([(x+.5-w/2)/(w/2),(y+.5-h/2)/(h/2),np.ones_like(x)],-1).astype('f')
    R=V[:3,:3];C=-R.T@V[:3,3];world=rays@R
    p=C+world*depth[:,:,None]
    dx=p[:,2:]-p[:,:-2];dy=p[2:]-p[:-2]
    n=np.zeros_like(p);n[1:-1,1:-1]=np.cross(dx[1:-1],dy[:,1:-1]);n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-9)
    return p,n,world,C


def fit_planes(p,n,spacing,max_planes=20):
    import open3d as o3d
    o3d.utility.random.seed(721);planes=[];remaining=np.ones(len(p),bool)
    for k in range(max_planes):
        ids=np.flatnonzero(remaining)
        if len(ids)<1000:break
        cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p[ids]))
        plane,inliers=cloud.segment_plane(spacing*1.8,3,1000,probability=.999)
        plane=np.array(plane);keep=(abs(p@plane[:3]+plane[3])<spacing*1.8)&(abs(n@plane[:3])>.9)&remaining
        remaining[ids[np.asarray(inliers)]]=False
        if keep.sum()<max(600,len(p)*.005):continue
        points=p[keep];center=points.mean(0);_,_,axes=np.linalg.svd(points-center,full_matrices=False)
        normal=axes[-1];offset=-normal@center
        residual=abs(points@normal+offset);extent=np.ptp((points-center)@axes[:2].T,axis=0)
        remaining[keep]=False
        if np.min(extent)<spacing*15 or np.quantile(residual,.9)>spacing*1.5:continue
        duplicate=False
        for old in planes:
            dot=normal@old['normal']
            if abs(dot)>.998 and abs(offset-np.sign(dot)*old['offset'])<spacing*3:duplicate=True
        if duplicate:continue
        planes.append(dict(normal=normal.tolist(),offset=float(offset),support=int(keep.sum()),extent=extent.tolist(),residual90=float(np.quantile(residual,.9))))
        print('Plane',len(planes),planes[-1],flush=True)
    return planes


def consolidate(depth,alpha,V,planes,tolerance):
    from scipy.ndimage import label
    p,n,rays,C=backproject(depth,V);best=np.full(depth.shape,np.inf,'f');output=depth.copy();owner=np.full(depth.shape,-1,'i')
    valid=np.isfinite(depth)&(depth>.1)&(depth<30)&(alpha>.95)
    for i,plane in enumerate(planes):
        normal=np.array(plane['normal'],dtype='f');offset=plane['offset'];dist=abs(p@normal+offset);alignment=abs(n@normal)
        denom=rays@normal;new=-(C@normal+offset)/np.where(abs(denom)>1e-6,denom,1e-6)
        cost=dist+.5*tolerance*(1-alignment)
        gate=valid&(dist<tolerance)&(alignment>.94)&(new>.1)&(new<30)&(abs(new-depth)<tolerance*3)&(cost<best)
        groups,count=label(gate);sizes=np.bincount(groups.ravel());gate&=sizes[groups]>=64
        output[gate]=new[gate];best[gate]=cost[gate];owner[gate]=i
    return output,owner


def run(a):
    meta=json.loads((a.source/'capture.json').read_text('utf8'));z=np.load(a.mesh);p=z['p'];tri=z['tri'];spacing=float((np.array(meta['bounds'])[1]-meta['bounds'][0]).max()/640)
    rng=np.random.default_rng(721);t=tri[rng.choice(len(tri),min(140000,len(tri)),replace=False)];points=p[t].mean(1);normals=np.cross(p[t[:,1]]-p[t[:,0]],p[t[:,2]]-p[t[:,0]]);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-9)
    planes=fit_planes(points,normals,spacing)
    a.out.mkdir(parents=True,exist_ok=True)
    for name in ['scene.json','aligned-cameras.npz']:shutil.copyfile(a.source/name,a.out/name)
    (a.out/'data-512').mkdir(exist_ok=True);shutil.copyfile(a.source/'data-512/dataset.json',a.out/'data-512/dataset.json')
    records=[];camera_support=[set() for _ in planes]
    # Plane support must occur in multiple ORIGINAL station depth maps.
    for f in meta['frames']:
        if not f['original_camera']:continue
        z=np.load(a.source/f['depth']);d=z['depth'][::4,::4];aa=z['alpha'][::4,::4]
        _,owner=consolidate(d,aa,np.array(f['viewmat']),planes,spacing*2)
        for i in range(len(planes)):
            if (owner==i).sum()>128:camera_support[i].add(Path(f['depth']).parent.name)
    accepted=[i for i,s in enumerate(camera_support) if len(s)>=min(3,len({Path(f['depth']).parent.name for f in meta['frames'] if f['original_camera']}))]
    planes=[dict(planes[i],supporting_stations=sorted(camera_support[i])) for i in accepted]
    for idx,f in enumerate(meta['frames']):
        z=np.load(a.source/f['depth']);d=z['depth'];aa=z['alpha'];new,owner=consolidate(d,aa,np.array(f['viewmat']),planes,spacing*2)
        out=a.out/f['depth'];out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,depth=new,alpha=aa)
        shutil.copyfile(a.source/f['color'],a.out/f['color']);records.append(dict(frame=f['depth'],changed=int((owner>=0).sum()),max_depth_change=float(abs(new-d).max())))
        if idx%12==0:print('Planar depth',idx,'/',len(meta['frames']),flush=True)
    meta.update(planar_consolidation=True,input_capture_sha256=hashlib.sha256((a.source/'capture.json').read_bytes()).hexdigest())
    (a.out/'capture.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    (a.out/'planes.json').write_text(json.dumps(dict(planes=planes,spacing=spacing,records=records,no_unknown_depth_filled=True),indent=2),encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--mesh',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
