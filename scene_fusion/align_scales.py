"""Align each native PLY to photo bundle cameras; estimate scale from training tracks."""
import argparse
from pathlib import Path
import cv2 as cv
import numpy as np
from core import load,prep,write_json
import json

def run(out):
    scene=load(out);reg=json.loads((out/'registration.json').read_text());bundle=np.load(out/'photo-bundle/bundle.npz')
    C,Q=bundle['C'],bundle['Q'];tracks=json.loads((out/'photo-bundle/tracks.json').read_text());train=bundle['train'];P=bundle['points']
    matrices=[];reports=[]
    for i,item in enumerate(scene['stations']):
        folder=out/item['id'];cam=np.load(folder/'cache.npz');f=np.load(out/'photo-bundle'/f"features-{item['id']}.npz")
        rays=[];targets=[]
        for k,track_id in enumerate(train):
            for camera,feature in tracks[track_id]:
                if camera==i:rays.append(f['ray'][feature]);targets.append(P[k])
        rays=np.array(rays);targets=np.array(targets);depth=np.zeros(len(rays));valid=np.zeros(len(rays),bool)
        face=np.argmax(np.array([rays@b[:,2] for b in prep.FACES]),axis=0)
        for k,b in enumerate(prep.FACES):
            ids=np.flatnonzero(face==k);q=rays[ids]@b;xy=(q[:,:2]/q[:,2,None]*320+320).astype('f');z=np.load(folder/f'depth-{k}.npz')
            dep=cv.remap(z['depth'],xy[:,0,None],xy[:,1,None],cv.INTER_LINEAR)[:,0]
            alpha=cv.remap(z['alpha'],xy[:,0,None],xy[:,1,None],cv.INTER_LINEAR)[:,0]
            depth[ids]=dep/q[:,2];valid[ids]=(alpha>.95)&(dep>.1)
        worldray=rays@Q[i].T;targetdepth=((targets-C[i])*worldray).sum(1)
        direction=(targets-C[i])/np.linalg.norm(targets-C[i],axis=1)[:,None]
        valid&=(np.linalg.norm(direction-worldray,axis=1)<.015)&(targetdepth>0)
        oldscale=np.cbrt(np.linalg.det(np.array(reg['matrices'][i])[:3,:3]))
        ratios=targetdepth[valid]/depth[valid];ratios=ratios[(ratios>oldscale*.5)&(ratios<oldscale*2)]
        if len(ratios)<12:raise ValueError('Insufficient depth scale evidence for '+item['id'])
        scale=float(np.median(ratios));A=scale*Q[i]@cam['R'];m=np.eye(4);m[:3,:3]=A;m[:3,3]=C[i]-A@cam['C'];matrices.append(m)
        keep=valid&(targetdepth>depth*scale*.7)&(targetdepth<depth*scale*1.4)
        np.savez_compressed(folder/'depth-anchors.npz',points=C[i]+worldray[keep]*(depth[keep]*scale)[:,None],
            log_ratio=np.log(targetdepth[keep]/(depth[keep]*scale)),target=targets[keep])
        reports.append(dict(id=item['id'],samples=len(ratios),initial_scale=oldscale,scale=scale,
            relative_spread=float(np.quantile(abs(ratios-scale)/scale,.9))))
    np.savez(out/'aligned-cameras.npz',C=C,Q=Q,matrices=np.array(matrices))
    write_json(out/'scales.json',reports);print(json.dumps(reports),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
