"""Source-camera photo recoloring ablations with unchanged Gaussian geometry.

Ownership is explicit; cameras never paint through an arbitrary nearest-view
projection. A conservative variant uses native-depth and photo-flatness gates.
These are appearance experiments, not hidden-surface reconstruction.
"""
import argparse
from pathlib import Path
import numpy as np,cv2 as cv
from scipy.spatial.transform import Rotation
from core import load,write_json
from surface_volume import fields

SH0=.2820947918

def sample_erp(image,points,center,basis):
    q=(points-center)@basis;u=np.arctan2(-q[:,1],q[:,0])/(2*np.pi)+.5
    v=np.arctan2(np.linalg.norm(q[:,:2],axis=1),q[:,2])/np.pi
    result=[]
    for first in range(0,len(points),24000):
        x=(u[first:first+24000,None]*image.shape[1]-.5).astype('f')
        y=np.clip(v[first:first+24000,None]*image.shape[0]-.5,0,image.shape[0]-1).astype('f')
        result.append(cv.remap(image,x,y,cv.INTER_LINEAR,borderMode=cv.BORDER_WRAP)[:,0])
    return np.concatenate(result)

def recolor_weight(depth_delta,confidence,distance,photo_std):
    visible=(confidence>.5)&(abs(depth_delta)<.035+.025*distance)
    # Pixel variance controls blending only; never edit edges to make a metric
    # improve or fill missing geometry with a foreground photograph.
    flat=np.clip((.055-photo_std)/.035,0,1)
    return visible*flat

def footprint_span(image,gaussians,center,basis):
    """Color variation over a two-sigma footprint, not only its center pixel."""
    rgb=sample_erp(image,gaussians[:,:3],center,basis)/255.
    lo=rgb.copy();hi=rgb.copy()
    frame=Rotation.from_quat(gaussians[:,[11,12,13,10]]).as_matrix()
    for axis in range(3):
        offset=2*np.exp(gaussians[:,7+axis,None])*frame[:,:,axis]
        for sign in [-1,1]:
            color=sample_erp(image,gaussians[:,:3]+sign*offset,center,basis)/255.
            lo=np.minimum(lo,color);hi=np.maximum(hi,color)
    return (hi-lo).max(1)

def run(scene,model,owners,protected,out,restore_from=None):
    out.mkdir(parents=True,exist_ok=True);meta=load(scene);cams=np.load(scene/'aligned-cameras.npz');C,Q=cams['C'],cams['Q']
    a=np.load(model);owner=np.load(owners);frozen=np.load(protected) if protected else np.zeros(len(a),bool)
    if a.ndim!=2 or a.shape[1]!=14 or not np.isfinite(a).all():raise ValueError('Invalid Gaussian array')
    if owner.shape!=(len(a),) or not np.issubdtype(owner.dtype,np.integer) or frozen.shape!=(len(a),) or frozen.dtype!=bool:raise ValueError('Invalid ownership/protection masks')
    if np.any(owner<0) or np.any(owner>=len(C)):raise ValueError('Invalid source camera index')
    direct=a.copy();guarded=a.copy();footprint=a.copy();views=fields(scene);weights=np.zeros(len(a),'f');reports=[]
    for i,item in enumerate(meta['stations']):
        ids=np.flatnonzero((owner==i)&~frozen)
        if not len(ids):continue
        image=cv.imdecode(np.fromfile(item['photo'],np.uint8),1)[:,:,::-1].copy()
        rgb=sample_erp(image,a[ids,:3],C[i],Q[i])/255.
        small=cv.resize(image,(3840,1920)).astype('f')/255
        mean=cv.boxFilter(small,-1,(11,11));variance=np.maximum(0,cv.boxFilter(small*small,-1,(11,11))-mean*mean)
        std=np.sqrt(variance.max(2));sampled_std=sample_erp(std,a[ids,:3],C[i],Q[i])
        delta,confidence,distance=views[i].sample(a[ids,:3]);w=recolor_weight(delta,confidence,distance,sampled_std);weights[ids]=w
        current=np.clip(a[ids,3:6]*SH0+.5,0,1)
        direct[ids,3:6]=(rgb-.5)/SH0;guarded[ids,3:6]=(current*(1-w[:,None])+rgb*w[:,None]-.5)/SH0
        span=footprint_span(image,a[ids],C[i],Q[i]);fw=w*np.clip((.08-span)/.05,0,1)
        footprint[ids,3:6]=(current*(1-fw[:,None])+rgb*fw[:,None]-.5)/SH0
        error=np.linalg.norm(current-rgb,axis=1)
        report=dict(camera=item['id'],points=len(ids),recolored=int((w>.5).sum()),footprint_recolored=int((fw>.5).sum()),
            eligible_flat_color_error_median=float(np.median(error[w>.5])) if (w>.5).any() else None,
            large_color_error_fraction=float((error>.15).mean()))
        reports.append(report);print(report,flush=True)
    np.save(out/'direct.npy',direct);np.save(out/'guarded.npy',guarded);np.save(out/'weights.npy',weights)
    np.save(out/'footprint-guarded.npy',footprint)
    if restore_from:
        from scipy.spatial import cKDTree
        base=np.load(restore_from/'fit-hd/optimized.npy');original=np.load(restore_from/'initial.npy');ids=np.flatnonzero(~frozen)
        dist,index=cKDTree(base[:,:3]).query(a[ids,:3],workers=4)
        if dist.max()>1e-7:raise ValueError('Cannot restore colors: input points do not match baseline identities')
        if not np.array_equal(a[ids],base[index]):raise ValueError('Ambiguous baseline point identities; cannot isolate an ablation')
        restored=a.copy();restored[ids,3:6]=original[index,3:6];np.save(out/'native-colors.npy',restored)
        native_geometry=a.copy();native_geometry[ids,:3]=original[index,:3];native_geometry[ids,6:]=original[index,6:]
        np.save(out/'native-geometry.npy',native_geometry)
        native_geometry[ids,3:6]=original[index,3:6];np.save(out/'native-all.npy',native_geometry)
    for candidate in [direct,guarded,footprint]:
        assert np.array_equal(a[:,:3],candidate[:,:3]) and np.array_equal(a[:,6:],candidate[:,6:])
        assert np.array_equal(a[frozen],candidate[frozen])
    write_json(out/'recoloring.json',dict(sources=reports,geometry_unchanged=True,opacity_unchanged=True,
        protected_points=int(frozen.sum()),semantic_masks=False,point_count=len(a),cameras=len(C)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--owners',type=Path,required=True);p.add_argument('--protected',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--restore-from',type=Path)
    a=p.parse_args();run(a.scene,a.model,a.owners,a.protected,a.out,a.restore_from)
