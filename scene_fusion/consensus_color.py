"""Conservative multi-view photo-guided diffuse color consensus experiment.

Changes appearance only where at least two depth-supported views agree that
the entire Gaussian footprint has almost constant color. This is not mesh
reconstruction, a spatial blur, or a remedy for wrong occlusion geometry.
"""
import argparse
from pathlib import Path
import numpy as np,cv2 as cv
from core import load,write_json
from surface_volume import fields
from photo_recolor import sample_erp,footprint_span,SH0

def run(a):
    a.out.mkdir(parents=True,exist_ok=True)
    z=np.load(a.model);base=z['base'].copy();sh=z['sh'].copy()
    meta=load(a.scene);cams=np.load(a.scene/'aligned-cameras.npz');depth=fields(a.scene)
    colors=np.full((len(meta['stations']),len(base),3),np.nan,'f');weights=np.zeros(colors.shape[:2],'f')
    for i,s in enumerate(meta['stations']):
        delta,confidence,distance=depth[i].sample(base[:,:3])
        ids=np.flatnonzero((confidence>.8)&(abs(delta)<.025+.015*distance))
        image=cv.imdecode(np.fromfile(s['photo'],np.uint8),1)[:,:,::-1].copy()
        # Original photograph is the range/edge guide; never the noisy splat colors.
        span=footprint_span(image,base[ids],cams['C'][i],cams['Q'][i])
        ids=ids[span<.025]
        rgb=sample_erp(image,base[ids,:3],cams['C'][i],cams['Q'][i])/255.
        colors[i,ids]=rgb;weights[i,ids]=confidence[ids]/np.maximum(distance[ids],.5)**2
        print('view',i+1,'flat supported',len(ids),flush=True)
    count=(weights>0).sum(0);eligible=count>=2;ids=np.flatnonzero(eligible)
    c=colors[:,ids];valid=weights[:,ids]>0
    lo=np.where(valid[:,:,None],c,np.inf).min(0);hi=np.where(valid[:,:,None],c,-np.inf).max(0)
    ids=ids[(hi-lo).max(1)<.035]
    w=weights[:,ids];values=np.nan_to_num(colors[:,ids]);target=(values*w[:,:,None]).sum(0)/w.sum(0)[:,None]
    # Fade spatial/photometric evidence at the confidence thresholds; keep SH
    # elsewhere. Do not let sparse views invent hue variation on agreed diffuse areas.
    span=(np.where(weights[:,ids,None]>0,colors[:,ids],-np.inf).max(0)-np.where(weights[:,ids,None]>0,colors[:,ids],np.inf).min(0)).max(1)
    blend=np.clip((.035-span)/.015,0,1)
    old=.5+SH0*sh[ids,0]
    sh[ids,0]=((1-blend[:,None])*old+blend[:,None]*target-.5)/SH0
    sh[ids,1:]*=(1-blend[:,None,None])
    base[ids,3:6]=sh[ids,0]
    np.savez(a.out/'optimized.npz',base=base,sh=sh)
    np.save(a.out/'selected.npy',ids)
    write_json(a.out/'consensus.json',dict(points=len(base),changed=len(ids),full_consensus=int((blend>.999).sum()),sources=len(meta['stations']),geometry_unchanged=True,opacity_unchanged=True,thresholds=dict(footprint_span=.025,multi_view_span=.035,minimum_views=2),status='experimental; requires source and midpoint validation'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
