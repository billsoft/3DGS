"""Keep virtual-camera depths only when original-station depths support them.

This tests agreement of synthetic depths; it cannot certify real-world geometry.
Occlusion by a foreground surface is unknown, not an automatic contradiction.
"""
import argparse,json,shutil,os
from pathlib import Path
import numpy as np
import cv2 as cv
from planar_depth import backproject

def classify_depth(projected,reference,valid,tolerance):
    agreement=valid&(abs(projected-reference)<=tolerance)
    free_space=valid&(projected<reference-tolerance)
    return agreement,free_space

def run(a):
    if a.out.exists():raise ValueError('Use a fresh candidate directory')
    meta=json.loads((a.source/'capture.json').read_text('utf8'));groups={}
    for f in meta['frames']:
        if f['original_camera']:groups.setdefault(str(Path(f['depth']).parent),[]).append(f)
    if not groups:raise ValueError('Original camera depths required')
    centers=[];stations=[]
    for frames in groups.values():
        V=np.array(frames[0]['viewmat']);centers.append(-V[:3,:3].T@V[:3,3]);stations.append(frames)
    cache={};a.out.mkdir(parents=True)
    for name in ('scene.json','aligned-cameras.npz','data-512/dataset.json'):
        dst=a.out/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a.source/name,dst)
    records=[];spacing=float(np.ptp(np.asarray(meta['bounds']),axis=0).max()/512)
    for i,f in enumerate(meta['frames']):
        src=a.source/f['depth'];dst=a.out/f['depth'];dst.parent.mkdir(parents=True,exist_ok=True)
        if f['original_camera']:
            try:os.link(src,dst)
            except OSError:shutil.copyfile(src,dst)
        else:
            z=np.load(src);d=z['depth'];aa=z['alpha'].copy();V=np.array(f['viewmat']);points,_,_,C=backproject(d,V)
            nearest=np.argsort(np.linalg.norm(np.array(centers)-C,axis=1))[:min(a.neighbors,len(stations))]
            votes=np.zeros(d.shape,'u1');contradictions=votes.copy()
            for station in nearest:
                station_agree=np.zeros(d.shape,bool);station_free=station_agree.copy()
                for ref in stations[station]:
                    key=ref['depth']
                    if key not in cache:
                        r=np.load(a.source/key);cache[key]=(r['depth'].astype('f'),r['alpha'].astype('f'))
                    rd,ra=cache[key];R=np.array(ref['viewmat']);q=points@R[:3,:3].T+R[:3,3];h,w=rd.shape
                    zz=q[:,:,2];safe=np.maximum(zz,1e-6)
                    x=(q[:,:,0]/safe*(w/2)+w/2-.5).astype('f');y=(q[:,:,1]/safe*(h/2)+h/2-.5).astype('f')
                    sampled=cv.remap(rd,x,y,cv.INTER_NEAREST,borderMode=cv.BORDER_CONSTANT,borderValue=0)
                    alpha=cv.remap(ra,x,y,cv.INTER_NEAREST,borderMode=cv.BORDER_CONSTANT,borderValue=0)
                    valid=(zz>.1)&(x>=1)&(x<w-2)&(y>=1)&(y<h-2)&(alpha>=.95)&(sampled>.1)&(sampled<30)
                    agree,free=classify_depth(zz,sampled,valid,2*spacing+.015*sampled)
                    station_agree|=agree;station_free|=free
                votes+=station_agree;contradictions+=station_free&~station_agree
            original=(aa>=.95)&np.isfinite(d)&(d>.1)&(d<30)
            keep=(votes>=1)&(contradictions==0);aa[~keep]=0
            np.savez_compressed(dst,depth=d,alpha=aa)
            records.append(dict(frame=f['depth'],valid_before=int(original.sum()),valid_after=int((original&keep).sum()),unsupported=int((original&(votes==0)).sum()),contradicted=int((original&(contradictions>0)).sum())))
        color=a.out/f['color'];color.parent.mkdir(exist_ok=True,parents=True)
        try:os.link(a.source/f['color'],color)
        except OSError:shutil.copyfile(a.source/f['color'],color)
        if i%12==0:print('Source support',i,'/',len(meta['frames']),flush=True)
    meta['virtual_depth_support']=dict(neighbors=a.neighbors,absolute_tolerance=2*spacing,relative_tolerance=.015,original_depths_unchanged=True)
    (a.out/'capture.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    (a.out/'support.json').write_text(json.dumps(dict(records=records,**meta['virtual_depth_support']),indent=2),encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--neighbors',type=int,default=2)
    a=p.parse_args()
    if a.neighbors<1:p.error('neighbors must be positive')
    run(a)
