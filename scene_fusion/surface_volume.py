"""Scene-independent robust projective distance fusion from calibrated GS depths.

Depths are uncertain predictions, not RGB-D measurements. Keep camera-specific
evidence and explicit free-space votes; do not concatenate their surface layers.
"""
import argparse,json
from pathlib import Path
import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter,minimum_filter
from skimage.measure import marching_cubes
from core import load,prep,write_json

class DepthField:
    def __init__(self,root,item,C,Q,scale=1.):
        self.C=C;self.Q=Q;self.maps=[]
        for k in range(6):
            z=np.load(root/item['id']/f'depth-{k}.npz');d=z['depth'].astype('f')*scale;alpha=z['alpha']
            span=cv.dilate(d,np.ones((3,3),'u1'))-cv.erode(d,np.ones((3,3),'u1'))
            confidence=np.clip((.12-span/np.maximum(d,.1))/.10,0,1)*np.clip((alpha-.8)/.18,0,1)
            confidence[(d<.15)|~np.isfinite(d)]=0
            self.maps.append((d,confidence.astype('f')))
    def sample(self,p):
        q=(p-self.C)@self.Q;face=np.argmax(np.stack([q@b[:,2] for b in prep.FACES]),axis=0)
        diff=np.zeros(len(p),'f');weight=diff.copy();distance=np.linalg.norm(q,axis=1).astype('f')
        for k,b in enumerate(prep.FACES):
            ids=np.flatnonzero(face==k)
            for start in range(0,len(ids),24000):
                ix=ids[start:start+24000];v=q[ix]@b;z=np.maximum(v[:,2],1e-6);d,w=self.maps[k];size=d.shape[0]
                xy=((v[:,:2]/z[:,None]+1)*size/2-.5).astype('f')
                def sample(im):return cv.remap(im,xy[:,0,None],xy[:,1,None],cv.INTER_LINEAR,borderMode=cv.BORDER_REPLICATE)[:,0]
                diff[ix]=(sample(d)-z)*distance[ix]/z
                weight[ix]=sample(w)*(z>.15)
        return diff,weight,distance

def fields(source,warped=False):
    meta=load(source);cam=np.load(source/'aligned-cameras.npz')
    root=source/'warped-depth' if warped else source
    return [DepthField(root,item,cam['C'][i],cam['Q'][i],1. if warped else np.cbrt(np.linalg.det(cam['matrices'][i,:3,:3]))) for i,item in enumerate(meta['stations'])]

def run(source,out,resolution=320,warped=False,free_weight=.35,back_weight=.025):
    out.mkdir(parents=True,exist_ok=True);meta=load(source);views=fields(source,warped)
    sample=np.concatenate([np.load(source/('warped' if warped else 'aligned')/(s['id']+'.npy'),mmap_mode='r')[::100,:3] for s in meta['stations']])
    lo,hi=np.quantile(sample,[.005,.995],axis=0);spacing=float((hi-lo).max()/resolution);lo-=spacing*4;hi+=spacing*4
    dims=np.ceil((hi-lo)/spacing).astype(int)+1
    if np.prod(dims)>24_000_000:raise ValueError('Surface grid exceeds memory budget')
    trunc=spacing*4;volume=np.empty(dims,'f');valid=np.zeros(dims,bool);yz=np.indices(dims[1:]).reshape(2,-1).T
    print('Grid',dims,'spacing',spacing,'truncation',trunc,flush=True)
    for x in range(dims[0]):
        points=lo+spacing*np.c_[np.full(len(yz),x),yz];acc=np.zeros(len(points),'f');total=acc.copy();observed=np.zeros(len(points),'u1')
        for view in views:
            delta,w,distance=view.sample(points);observed+=(w>.25)&(abs(delta)<trunc*2)
            w=w/np.maximum(.5,distance)**1.1
            w*=np.where(delta>trunc,free_weight,np.where(delta<-trunc,back_weight,1.))
            acc+=w*np.clip(delta/trunc,-1,1);total+=w
        volume[x]=np.divide(acc,total,out=np.ones_like(acc),where=total>1e-7).reshape(dims[1:]);valid[x]=((total>1e-7)&(observed>0)).reshape(dims[1:])
        if x%40==0:print('Volume',x,'/',dims[0],flush=True)
    filtered=gaussian_filter(volume,.6);mask=minimum_filter(valid.astype('u1'),size=3)>0
    vertices,tri,_,_=marching_cubes(filtered,level=0,mask=mask,allow_degenerate=False)
    p=(vertices*spacing+lo).astype('f');np.savez_compressed(out/'surface.npz',p=p,tri=tri)
    np.savez_compressed(out/'volume.npz',field=filtered,valid=valid,origin=lo,spacing=spacing)
    report=dict(vertices=len(p),triangles=len(tri),spacing=spacing,truncation=trunc,grid=dims.tolist(),sources=len(views),warped=warped,free_weight=free_weight,back_weight=back_weight,semantic_rules=False)
    write_json(out/'surface-report.json',report);print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--resolution',type=int,default=320);p.add_argument('--warped',action='store_true');p.add_argument('--free-weight',type=float,default=.35);p.add_argument('--back-weight',type=float,default=.025);a=p.parse_args();run(a.source,a.out,a.resolution,a.warped,a.free_weight,a.back_weight)
