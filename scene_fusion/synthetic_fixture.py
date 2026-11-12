"""Independent textured room fixture, two cameras, differing units and Unicode names.

Only for repeatable integration testing; never presented as user-captured data.
"""
import argparse
from pathlib import Path
import numpy as np
import cv2 as cv
import torch
from scipy.spatial.transform import Rotation
from core import prep,gs,viewmat,write_json
from refit_render import load_scene,render

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(123);clouds=[]
for axis,coordinate,lo,hi in [(0,-3,[-2,0],[2,3]),(0,3,[-2,0],[2,3]),(1,-2,[-3,0],[3,3]),(1,2,[-3,0],[3,3]),(2,0,[-3,-2],[3,2]),(2,3,[-3,-2],[3,2])]:
    other=[j for j in range(3) if j!=axis];u,v=np.meshgrid(np.arange(lo[0],hi[0]+.02,.045),np.arange(lo[1],hi[1]+.02,.045));xyz=np.zeros((u.size,3));xyz[:,axis]=coordinate;xyz[:,other[0]]=u.ravel();xyz[:,other[1]]=v.ravel()
    tile=np.floor((np.c_[u.ravel(),v.ravel()]-lo)/.18).astype(int)
    palette=rng.uniform(.12,.92,size=(tile[:,0].max()+1,tile[:,1].max()+1,3));rgb=palette[tile[:,0],tile[:,1]]
    normal=np.eye(3)[:,axis];t=np.eye(3)[:,other[0]];frame=np.stack([normal,t,np.cross(normal,t)],axis=1);q=Rotation.from_matrix(frame).as_quat()[[3,0,1,2]]
    row=np.zeros((len(xyz),14),'f');row[:,:3]=xyz;row[:,3:6]=(rgb-.5)/.2820947918;row[:,6]=4;row[:,7:10]=np.log([.004,.032,.032]);row[:,10:14]=q;clouds.append(row)
world=np.concatenate(clouds);np.save(a.out/'world.npy',world);scene=load_scene(a.out/'world.npy')
centers=[np.array([-1.2,-.8,1.4]),np.array([1.,.7,1.4])];names=['机位 甲','view-final'];scales=[.01,100.]
H,W=1024,2048;yy,xx=np.indices((H,W));lon=((xx+.5)/W-.5)*2*np.pi;lat=(yy+.5)/H*np.pi
rays=np.stack([np.cos(lon)*np.sin(lat),-np.sin(lon)*np.sin(lat),np.cos(lat)],-1).reshape(-1,3)
face=np.argmax(np.array([rays@b[:,2] for b in prep.FACES]),axis=0)
for i,(center,name,scale) in enumerate(zip(centers,names,scales)):
    Q=Rotation.from_euler('z',i*.35).as_matrix();local=Rotation.from_euler('xyz',[.12*i,.07,-.3*i]).as_matrix();m=np.eye(4);m[:3,:3]=scale*local;m[:3,3]=-m[:3,:3]@center
    gs.write(a.out/(name+'.ply'),gs.transform(world,m));pixels=np.zeros((H*W,3),'u1')
    with torch.no_grad():
        for k,b in enumerate(prep.FACES):
            im,_,_=render(scene,dict(viewmat=viewmat(center,Q@b).tolist()),1024)
            image=(im.clamp(0,1).cpu().numpy()*255).astype('u1');ids=np.flatnonzero(face==k);r=rays[ids]@b;uv=((r[:,:2]/r[:,2,None]+1)*512-.5).astype('f')
            # OpenCV remap has a 32767-row limit.
            for start in range(0,len(ids),24000):
                ix=ids[start:start+24000];xy=uv[start:start+24000]
                pixels[ix]=cv.remap(image,xy[:,0,None],xy[:,1,None],cv.INTER_LINEAR,borderMode=cv.BORDER_REPLICATE)[:,0]
    ok,encoded=cv.imencode('.png',pixels.reshape(H,W,3)[:,:,::-1]);assert ok;encoded.tofile(a.out/(name+'.png'))
write_json(a.out/'fixture.json',dict(synthetic=True,points=len(world),centers=[c.tolist() for c in centers],input_scale_factors=scales,labels=names))
print('Synthetic fixture generated',len(world),'Gaussians; scale ratio',scales[1]/scales[0],flush=True)
