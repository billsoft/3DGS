"""Geometry-only, edge-protected smoothing before Gaussian-depth meshing.

Color is deliberately not a guide: appearance blotches must not become geometry.
Unknown depths stay unknown. This regularizes synthetic depth, not measured truth.
"""
import argparse,json,hashlib,shutil,os
from pathlib import Path
import numpy as np
import cv2 as cv

PROFILES={'medium':dict(radius=3,sigma=.012,iterations=2,max_change=.015),
          'strong':dict(radius=5,sigma=.025,iterations=4,max_change=.035)}

def smooth_depth(depth,alpha,profile='strong'):
    cfg=PROFILES[profile];d=np.asarray(depth,dtype=np.float32)
    valid=np.isfinite(d)&(d>.1)&(d<30)&(alpha>=.95)
    if not valid.any():return d.copy(),dict(valid=0,changed=0,protected=0)
    # Log depth makes the range kernel relative to distance, not scene scale.
    work=np.log(np.where(valid,d,np.median(d[valid]))).astype('f')
    robust=cv.medianBlur(work,3);edge=np.zeros(d.shape,'u1')
    limit=max(.06,3*cfg['sigma'])
    for axis in (0,1):
        a=[slice(None)]*2;b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1)
        a,b=tuple(a),tuple(b);jump=(abs(robust[a]-robust[b])>limit)&valid[a]&valid[b]
        edge[a]|=jump;edge[b]|=jump
    radius=cfg['radius'];kernel=np.ones((2*radius+1,2*radius+1),'u1')
    protected=cv.dilate(edge,kernel)>0
    protected|=cv.dilate((~valid).astype('u1'),kernel)>0
    # Do not invent observations just outside cubemap bounds.
    protected[:radius]=True;protected[-radius:]=True;protected[:,:radius]=True;protected[:,-radius:]=True
    active=valid&~protected;original=work.copy()
    for _ in range(cfg['iterations']):
        filtered=cv.bilateralFilter(work,2*radius+1,cfg['sigma'],radius*.65)
        work[active]=filtered[active]
        work=np.clip(work,original-np.log1p(cfg['max_change']),original+np.log1p(cfg['max_change']))
    result=d.copy();result[active]=np.exp(work[active])
    delta=abs(result[valid]-d[valid])/d[valid]
    def roughness(z):
        dd=abs(z[1:-1,2:]-2*z[1:-1,1:-1]+z[1:-1,:-2])+abs(z[2:,1:-1]-2*z[1:-1,1:-1]+z[:-2,1:-1])
        mask=active[1:-1,1:-1]
        return float(np.mean(dd[mask])) if mask.any() else None
    return result,dict(valid=int(valid.sum()),changed=int((valid&(abs(result-d)>1e-6)).sum()),protected=int((valid&protected).sum()),
        relative_change_p95=float(np.quantile(delta,.95)),relative_change_max=float(delta.max()),
        roughness_before=roughness(original),roughness_after=roughness(work))

def run(a):
    if a.out.exists():raise ValueError('Use a new output directory; preserve baselines')
    meta=json.loads((a.source/'capture.json').read_text('utf8'));a.out.mkdir(parents=True)
    for name in ('scene.json','aligned-cameras.npz','data-512/dataset.json'):
        target=a.out/name;target.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(a.source/name,target)
    records=[]
    for i,f in enumerate(meta['frames']):
        z=np.load(a.source/f['depth']);d,record=smooth_depth(z['depth'],z['alpha'],a.profile)
        target=a.out/f['depth'];target.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target,depth=d,alpha=z['alpha'])
        color=a.out/f['color'];color.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(a.source/f['color'],color)
        except OSError:shutil.copyfile(a.source/f['color'],color)
        records.append(dict(frame=f['depth'],**record))
        if i%12==0:print(a.profile,i,'/',len(meta['frames']),flush=True)
    report=dict(profile=a.profile,parameters=PROFILES[a.profile],records=records,no_unknown_depth_filled=True,
        alpha_unchanged=True,color_unchanged=True,input_capture_sha256=hashlib.sha256((a.source/'capture.json').read_bytes()).hexdigest())
    meta['geometry_smoothing']=dict(profile=a.profile,parameters=PROFILES[a.profile],synthetic=True)
    (a.out/'capture.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    (a.out/'smoothing.json').write_text(json.dumps(report,indent=2),encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--profile',choices=PROFILES,default='strong');run(p.parse_args())
