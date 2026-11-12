"""Project contributing Gaussians onto image-supported bounded planes.

Preserve each source ray and transport full covariance with the projection
Jacobian; retain foreground occluders and geometry outside each polygon.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from core import load,write_json

def plane_projection(points,center,corners,normal):
    ray=points-center;den=ray@normal;d=(corners[0]-center)@normal
    safe=np.where(abs(den)>1e-6,den,np.copysign(1e-6,den+1e-12));scale=d/safe;hit=center+scale[:,None]*ray
    u=corners[1]-corners[0];u/=np.linalg.norm(u);v=np.cross(normal,u);v/=np.linalg.norm(v);basis=np.stack([u,v],axis=1)
    polygon=(corners-corners[0])@basis;xy=(hit-corners[0])@basis;edge=np.roll(polygon,-1,axis=0)-polygon
    delta=xy[:,None]-polygon[None]
    cross=edge[None,:,0]*delta[:,:,1]-edge[None,:,1]*delta[:,:,0];inside=(cross>=-1e-6).all(1)|(cross<=1e-6).all(1)
    original=np.linalg.norm(ray,axis=1);target=original*scale
    # Relative to surface extent, not a room-specific distance in world units.
    foreground_margin=.02*np.linalg.norm(edge,axis=1).min()
    eligible=inside&(scale>.15)&(scale<4)&(original>=target-foreground_margin)
    J=scale[:,None,None]*(np.eye(3)[None]-ray[:,:,None]*normal[None,None,:]/safe[:,None,None])
    return hit,J,eligible,original-target

def run(source,out,planes_path):
    meta=load(source);C=np.load(source/'aligned-cameras.npz')['C'];planes=json.loads(planes_path.read_text(encoding='utf8'))['planes'];dest=out/'warped';dest.mkdir(parents=True,exist_ok=True);report=[]
    for i,item in enumerate(meta['stations']):
        a=np.load(source/'aligned'/(item['id']+'.npy'));result=a.copy();changed=np.zeros(len(a),bool);stats=[]
        for pi,plane in enumerate(planes):
            corners=np.array(plane['corners']);normal=np.array(plane['normal'])
            residuals=[]
            for start in range(0,len(a),30000):
                ids=np.arange(start,min(start+30000,len(a)));mapped,J,keep,residual=plane_projection(a[ids,:3],C[i],corners,normal);keep&=~changed[ids]
                ix=ids[keep]
                if not len(ix):continue
                residuals.extend(residual[keep].tolist());result[ix,:3]=mapped[keep]
                R=Rotation.from_quat(a[ix][:,[11,12,13,10]]).as_matrix();cov=(R*np.exp(2*a[ix,None,7:10]))@R.transpose(0,2,1)
                transformed=J[keep]@cov@J[keep].transpose(0,2,1)+.002**2*np.outer(normal,normal)[None]
                eig,frame=np.linalg.eigh(transformed);eig=np.maximum(eig,1e-8);frame[:,:,0]*=np.linalg.det(frame)[:,None]
                result[ix,7:10]=.5*np.log(eig);result[ix,10:14]=Rotation.from_matrix(frame).as_quat()[:,[3,0,1,2]];changed[ix]=True
            stats.append(dict(plane=pi,points=len(residuals),radial_depth_error_quantiles=np.quantile(residuals,[.1,.5,.9]).tolist() if residuals else []))
        np.save(dest/(item['id']+'.npy'),result);np.save(dest/(item['id']+'-protected.npy'),changed);report.append(dict(id=item['id'],modified=int(changed.sum()),planes=stats));print(report[-1],flush=True)
    write_json(out/'plane-correction.json',dict(sources=report,source_ray_preserved=True,covariance_transport=True,semantic_rules=False,constraint_file=str(planes_path)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--planes',type=Path,required=True);a=p.parse_args();run(a.source,a.out,a.planes)
