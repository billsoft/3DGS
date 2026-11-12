"""Assign one photo to each common surface patch, then sample thin Gaussians."""
import argparse
from pathlib import Path
import numpy as np,cv2 as cv
from scipy.spatial.transform import Rotation
from core import load,write_json,gs
from surface_volume import fields

def sample_photo(im,p,C,Q):
    q=(p-C)@Q;u=np.arctan2(-q[:,1],q[:,0])/(2*np.pi)+.5;v=np.arctan2(np.linalg.norm(q[:,:2],axis=1),q[:,2])/np.pi
    results=[]
    for start in range(0,len(p),24000):
        tx=(u[start:start+24000,None]*im.shape[1]-.5).astype('f');ty=(v[start:start+24000,None]*im.shape[0]-.5).astype('f')
        results.append(cv.remap(im,tx,ty,cv.INTER_LINEAR,borderMode=cv.BORDER_WRAP)[:,0,::-1]/255.)
    return np.concatenate(results) if results else np.empty((0,3))

def adjacency(tri):
    edges=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1);face=np.tile(np.arange(len(tri)),3)
    order=np.lexsort((edges[:,1],edges[:,0]));same=(edges[order[1:]]==edges[order[:-1]]).all(1)
    a,b=face[order[:-1][same]],face[order[1:][same]];neighbors=np.repeat(np.arange(len(tri))[:,None],3,axis=1);used=np.zeros(len(tri),'i')
    for left,right in zip(a,b):
        if used[left]<3:neighbors[left,used[left]]=right;used[left]+=1
        if used[right]<3:neighbors[right,used[right]]=left;used[right]+=1
    return neighbors

def run(source,out,spacing=.025,warped=False):
    meta=load(source);c=np.load(source/'aligned-cameras.npz');C,Q=c['C'],c['Q'];views=fields(source,warped);N=len(views)
    mesh=np.load(out/'surface.npz');p,tri=mesh['p'],mesh['tri'];xyz=p[tri];center=xyz.mean(1);normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]);area=np.linalg.norm(normal,axis=1);normal/=np.maximum(area[:,None],1e-10)
    scores=np.zeros((len(tri),N),'f');colors=np.zeros((len(tri),N,3),'f');images=[]
    for i,(item,view) in enumerate(zip(meta['stations'],views)):
        image=cv.imdecode(np.fromfile(item['photo'],np.uint8),1);images.append(image)
        diff,w,dist=view.sample(center);angle=abs(((C[i]-center)*normal).sum(1))/np.maximum(dist,1e-6)
        supported=(w>.05)&(abs(diff)<.20+.025*dist)
        scores[:,i]=supported*(.2+.8*angle)*w/np.maximum(.4,dist)**1.5*np.exp(-abs(diff)/(.12+.02*dist))
        colors[:,i]=sample_photo(image,center,C[i],Q[i]);print('Color evidence',i,flush=True)
    owner=scores.argmax(1);good=scores.max(1)>0;neighbors=adjacency(tri);unary=-np.log(np.maximum(scores,1e-20)/np.maximum(scores.max(1)[:,None],1e-20))
    for _ in range(5):
        costs=unary.copy()
        for i in range(N):
            for k in range(3):
                other=owner[neighbors[:,k]];difference=np.linalg.norm(colors[:,i]-colors[np.arange(len(tri)),other],axis=1)
                costs[:,i]+=(other!=i)*(.15+np.minimum(difference,.5)*2)
        owner=costs.argmin(1)
    np.savez_compressed(out/'surface-ownership.npz',owner=owner,scores=scores,good=good)
    arrays=[];sources=[];face_ids=[]
    for i in range(N):
        ids=np.flatnonzero(good&(owner==i)&(area>1e-10));x=xyz[ids]
        longest=np.max(np.linalg.norm(x-np.roll(x,1,axis=1),axis=2),axis=1);subdivision=np.clip(np.ceil(longest/spacing).astype(int),1,16)
        for n in np.unique(subdivision):
            selected=np.flatnonzero(subdivision==n);u,v=np.indices((n,n));mask=u+v<n;uu,vv=u[mask],v[mask]
            bary=[np.c_[1-(uu+vv+2/3)/n,(uu+1/3)/n,(vv+1/3)/n]];mask=u+v<n-1;uu,vv=u[mask],v[mask]
            if len(uu):bary.append(np.c_[1-(uu+vv+4/3)/n,(uu+2/3)/n,(vv+2/3)/n])
            bary=np.vstack(bary)
            for start in range(0,len(selected),max(1,24000//len(bary))):
                sel=selected[start:start+max(1,24000//len(bary))];triangle=x[sel];q=triangle-triangle.mean(1)[:,None]
                cov=np.einsum('nki,nkj->nij',q,q)*(.4/n**2);eig,axes=np.linalg.eigh(cov);eig[:,0]=(.06*spacing)**2;eig[:,1:]=np.maximum(eig[:,1:],(.06*spacing)**2);axes[:,:,0]*=np.linalg.det(axes)[:,None]
                pos=np.einsum('bk,nkj->nbj',bary,triangle).reshape(-1,3);rgb=sample_photo(images[i],pos,C[i],Q[i]);a=np.empty((len(pos),14),'f')
                a[:,:3]=pos;a[:,3:6]=(rgb-.5)/.2820947918;a[:,6]=4.6;a[:,7:10]=np.repeat(.5*np.log(eig),len(bary),axis=0);a[:,10:14]=np.repeat(Rotation.from_matrix(axes).as_quat()[:,[3,0,1,2]],len(bary),axis=0)
                arrays.append(a);sources.append(np.full(len(a),i,'i2'));face_ids.append(np.repeat(ids[sel],len(bary)))
        print('Sampled source',i,flush=True)
    a=np.concatenate(arrays);np.save(out/'initial.npy',a);np.save(out/'source.npy',np.concatenate(sources));np.save(out/'surface-face.npy',np.concatenate(face_ids))
    gs.write(out/'initial.ply',a);gs.to_splat(out/'initial.splat',a);up=Q[:,:,2].mean(0);up/=np.linalg.norm(up)
    write_json(out/'fusion.json',dict(stations=[s['id'] for s in meta['stations']],station_labels=[s['label'] for s in meta['stations']],centers=C.tolist(),up=up.tolist(),splats=len(a),method='Common projective distance surface; coherent photo ownership; thin surface Gaussians',experimental=True))
    write_json(out/'surface-export.json',dict(points=len(a),faces=len(tri),unobserved_faces=int((~good).sum()),source_counts=np.bincount(np.concatenate(sources),minlength=N).tolist(),sampling_spacing=spacing))
    print('Surface Gaussians',len(a),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--spacing',type=float,default=.025);p.add_argument('--warped',action='store_true');a=p.parse_args();run(a.source,a.out,a.spacing,a.warped)
