"""Shared calibrated cameras and original inputs; no semantic object rules."""
from pathlib import Path
import sys,json,os
import cv2 as cv,numpy as np
ROOT=Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
import room_mesh as rm
OUT=ROOT/'four-room/fusion-next';OUT.mkdir(exist_ok=True)
BASE=ROOT/'four-room/surface-fusion-v3'
ba=np.load(ROOT/'four-room/photo-bundle/bundle.npz');C=ba['C'];Q=ba['Q'];stations=rm.stations
ds=json.loads((ROOT/'four-room/photo-bundle/depth-report.json').read_text())
scales=[];cams=[]
for i,n in enumerate(stations):
    cc=np.load(rm.room/n/'cache.npz');scale=np.cbrt(np.linalg.det(rm.matrices[i,:3,:3]))*ds[n]['scale_correction'];scales.append(scale)
    A=scale*Q[i]@cc['R'];m=np.eye(4);m[:3,:3]=A;m[:3,3]=C[i]-A@cc['C'];cams.append(rm.Camera(n,m))
def photos(size=None):
    a=[cv.imdecode(np.fromfile(rm.inputs/f'{n}_7680.jpg',np.uint8),1) for n in stations]
    return [cv.resize(im,size,interpolation=cv.INTER_AREA) for im in a] if size else a
def uv(q):
    return np.c_[np.arctan2(-q[:,1],q[:,0])/(2*np.pi)+.5,np.arctan2(np.linalg.norm(q[:,:2],axis=1),q[:,2])/np.pi].astype('f')
def sample(im,tx):
    if not len(tx):return np.empty((0,)+im.shape[2:],dtype=im.dtype)
    out=[]
    for j in range(0,len(tx),24000):
        t=tx[j:j+24000];out.append(cv.remap(im,(t[:,0,None]*im.shape[1]-.5).astype('f'),(t[:,1,None]*im.shape[0]-.5).astype('f'),cv.INTER_LINEAR,borderMode=cv.BORDER_WRAP)[:,0])
    return np.concatenate(out)
def project(p,i):return (p-C[i])@Q[i]
def correct_rgb(rgb,i):
    path=OUT/'appearance-report.json'
    if not path.exists():return rgb
    params=json.loads(path.read_text());return np.clip(rgb*np.array(params['gain'])[i]+np.array(params['offset'])[i],0,1)
def write_json(name,data):(OUT/name).write_text(json.dumps(data,indent=2),encoding='utf8')
def mesh():
    meta=json.loads((BASE/'mesh.json').read_text());rows=np.fromfile(BASE/'mesh.bin','<f4').reshape(-1,3,5)
    # All surfaces, including previous successful priors, enter the same optimizer.
    p,tri=np.unique(rows[:,:,:3].reshape(-1,3),axis=0,return_inverse=True)
    tri=tri.reshape(-1,3);prior_uv=rows[:,:,3:].copy();prior_group=np.repeat(np.arange(len(meta['groups'])),[g['count']//3 for g in meta['groups']])
    # Conforming edge subdivision prevents a large triangle's center from
    # deciding visibility/ownership for an entire table or wall.
    for iteration in range(10):
        edge=np.stack([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]],1)
        split=np.linalg.norm(p[edge[:,:,0]]-p[edge[:,:,1]],axis=2)>.12
        affected=split.any(1)
        if not affected.any():break
        marked=np.unique(np.sort(edge[split],axis=1),axis=0);lookup={tuple(e):len(p)+k for k,e in enumerate(marked)}
        p=np.vstack([p,(p[marked[:,0]]+p[marked[:,1]])*.5]);new=[];newuv=[];newgroup=[]
        for face,flags,tx,group in zip(tri[affected],split[affected],prior_uv[affected],prior_group[affected]):
            mids=[lookup.get(tuple(sorted((face[k],face[(k+1)%3]))),-1) for k in range(3)];count=int(flags.sum())
            mapping={int(vertex):value for vertex,value in zip(face,tx)}
            for k in range(3):
                if mids[k]>=0:mapping[mids[k]]=(tx[k]+tx[(k+1)%3])*.5
            if count==1:
                k=int(np.flatnonzero(flags)[0]);a,b,c=face[[k,(k+1)%3,(k+2)%3]];m=mids[k];children=[[a,m,c],[m,b,c]]
            elif count==2:
                k=next(k for k in range(3) if flags[k] and flags[(k-1)%3]);a,b,c=face[[k,(k+1)%3,(k+2)%3]];u=mids[k];v=mids[(k-1)%3];children=[[a,u,v],[u,b,c],[u,c,v]]
            else:
                a,b,c=face;u,v,w=mids;children=[[a,u,w],[u,b,v],[w,v,c],[u,v,w]]
            new.extend(children);newuv.extend([[mapping[int(v)] for v in child] for child in children]);newgroup.extend([group]*len(children))
        tri=np.vstack([tri[~affected],np.asarray(new,dtype='i')])
        prior_uv=np.concatenate([prior_uv[~affected],np.asarray(newuv)]);prior_group=np.r_[prior_group[~affected],newgroup]
    np.savez_compressed(OUT/'prior.npz',uv=prior_uv,group=prior_group)
    return p,tri,meta
def adjacency(tri):
    edges=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),1);fi=np.tile(np.arange(len(tri)),3)
    ix=np.lexsort(edges.T);edges=edges[ix];fi=fi[ix];hit=(edges[:-1]==edges[1:]).all(1)
    return fi[:-1][hit],fi[1:][hit]
