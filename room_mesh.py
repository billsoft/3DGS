"""Fuse calibrated collider depths and create a textured room mesh."""
import json,numpy as np,cv2 as cv
from pathlib import Path
from scipy.ndimage import gaussian_filter,minimum_filter
from scipy.spatial.transform import Rotation
from skimage.measure import marching_cubes
from collider_audit import f,room,reg,meta,inputs,read_mesh

OUT=room/'mesh'
OUT.mkdir(exist_ok=True)
matrices=np.array(reg['matrices'])
centers=np.array(meta['centers'])
stations=reg['stations']
class Camera:
    def __init__(self,n,m):
        cam=np.load(room/n/'cache.npz')
        self.R=cam['R'];self.C=cam['C'];self.inv=np.linalg.inv(m)
        self.scale=np.cbrt(np.linalg.det(m[:3,:3]))
        self.maps=[]
        for k in range(6):
            d=np.load(room/'colliders'/f'{n}-{k}.npy').astype('f')
            ker=np.ones((3,3),np.uint8)
            span=cv.dilate(d,ker)-cv.erode(d,ker)
            conf=np.clip((.16-span/np.maximum(.25,d))/.13,0,1)
            conf[(d>.999e8)|(d<.12)]=0
            self.maps.append((d,conf))
    def project(self,p):
        return ((p@self.inv[:3,:3].T+self.inv[:3,3])-self.C)@self.R.T
    def sample(self,q):
        primary=np.argmax(abs(q),axis=1)
        # Cubemap axis order: +X, -X, +Y, -Y, +Z, -Z.
        face=2*primary+(q[np.arange(len(q)),primary]<0)
        depth=np.zeros(len(q),'f');confidence=np.zeros(len(q),'f');delta=np.zeros(len(q),'f')
        for k,b in enumerate(f.prep.FACES):
            ids=np.flatnonzero(face==k)
            r=q[ids]@b
            uv=(320+320*r[:,:2]/np.maximum(r[:,2,None],1e-6)).astype('i').clip(0,639)
            d,w=self.maps[k];obs=d[uv[:,1],uv[:,0]]
            factor=np.linalg.norm(r,axis=1)/np.maximum(.0001,r[:,2])*self.scale
            delta[ids]=(obs-r[:,2])*factor
            confidence[ids]=w[uv[:,1],uv[:,0]]
            depth[ids]=np.linalg.norm(q[ids],axis=1)*self.scale
        return delta,confidence,depth
cameras=[Camera(n,m) for n,m in zip(stations,matrices)]
def volume():
    box=[]
    for n,m in zip(stations,matrices):
        p,_,_=read_mesh(inputs/f'{n}m.ply')
        box.append(p[::9]@m[:3,:3].T+m[:3,3])
    lo,hi=np.quantile(np.vstack(box),[.001,.999],axis=0)
    spacing=.065
    lo-=.22;hi+=.22
    dims=(np.ceil((hi-lo)/spacing)+1).astype(int)
    if dims.prod()>24_000_000:raise ValueError(f'Mesh grid exceeds capacity: {dims}')
    print('volume shape',dims,'bounds',lo,hi,flush=True)
    field=np.zeros(dims,'f');valid=np.zeros(dims,bool)
    yz=np.indices(dims[1:]).reshape(2,-1).T
    for x in range(dims[0]):
        p=lo+spacing*np.c_[np.full(len(yz),x),yz]
        accum=np.zeros(len(p),'f');total=np.zeros(len(p),'f')
        for camera in cameras:
            diff,w,dep=camera.sample(camera.project(p))
            trunc=.165
            w=w/np.maximum(.7,dep)**1.1
            w*=np.where(diff<-trunc,.065,np.where(diff>trunc,.27,1))
            accum+=w*np.clip(diff/trunc,-1,1);total+=w
        good=total>1e-5
        field[x]=np.divide(accum,total,out=np.zeros_like(total),where=good).reshape(dims[1:])
        valid[x]=good.reshape(dims[1:])
    # A light filter on supported values only, before extracting a zero crossing.
    weights=gaussian_filter(valid.astype('f'),.62)
    field=np.divide(gaussian_filter(field,.62),weights,out=np.ones_like(field),where=weights>.01)
    mask=minimum_filter(valid.astype('u1'),size=3)>0
    verts,tri,_,_=marching_cubes(field,0,mask=mask,allow_degenerate=False)
    p=(verts*spacing+lo).astype('f')
    np.savez_compressed(OUT/'volume.npz',p=p,tri=tri)
    print('volume mesh',len(p),len(tri),flush=True)

if __name__=='__main__':volume()
