"""Experimental surface-oriented refinement using gsplat's 2DGS backend.

Photographs supervise appearance; normal consistency and ray distortion supervise
surface coherence. The result is an intermediate surfel representation for mesh
extraction, NOT an interchangeable standard 3D Gaussian PLY. No object templates.
"""
import argparse,json,time,hashlib,shutil
from pathlib import Path
import numpy as np
from core import write_json,load,prep,viewmat


def initialize(path):
    from scipy.spatial.transform import Rotation
    z=np.load(path);base=z['base'].copy();sh=z['sh'].copy()
    q=base[:,10:14];axes=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    order=np.argsort(base[:,7:10],axis=1)[:,::-1]
    axes=np.take_along_axis(axes,order[:,None,:],axis=2)
    axes[:,:,2]*=np.linalg.det(axes)[:,None]
    base[:,7:10]=np.take_along_axis(base[:,7:10],order,axis=1)
    base[:,10:14]=Rotation.from_matrix(axes).as_quat()[:,[3,0,1,2]]
    return base,sh


def render_surface(model,v,size):
    import torch
    from gsplat import rasterization_2dgs,spherical_harmonics
    V=torch.tensor(v['viewmat'],device='cuda',dtype=torch.float32)[None]
    f=size/(2*np.tan(np.deg2rad(v.get('fov',90))/2))
    K=torch.tensor([[[f,0,size/2],[0,f,size/2],[0,0,1]]],device='cuda',dtype=torch.float32)
    center=-(V[0,:3,:3].T@V[0,:3,3]);sh=model['sh']
    rgb=(.5+spherical_harmonics(int(sh.shape[1]**.5)-1,model['means']-center,sh)).clamp(0,1)
    return rasterization_2dgs(model['means'],model['quats'],model['logs'].exp(),model['opacity'].sigmoid(),
        rgb[None],V,K,size,size,near_plane=.1,far_plane=30,packed=False,render_mode='RGB+ED',distloss=True,depth_mode='median')


def train(a):
    import torch,cv2
    import torch.nn.functional as F
    from refit_render import load_scene,render,SH0
    from refit_train import structure_loss
    if (a.out/'optimized.npz').resolve()==a.model.resolve():raise ValueError('Surface candidate must not overwrite the source 3DGS model')
    if (a.out/'optimized.npz').exists() and not a.resume:raise ValueError('Use a fresh candidate directory or explicitly resume')
    torch.set_num_threads(4);torch.manual_seed(721);rng=np.random.default_rng(721)
    a.out.mkdir(parents=True,exist_ok=True)
    data=a.source/f'data-{a.size}';meta=json.loads((data/'dataset.json').read_text('utf8'));views=meta['views']
    target=np.load(data/'targets.npy',mmap_mode='r');ids=[i for i,v in enumerate(views) if v['source'] is not None and v['role']=='train']
    if not ids:raise ValueError('Source photo views missing')
    if a.resume:
        z=np.load(a.resume);base,sh=z['base'],z['sh']
    else:base,sh=initialize(a.model)
    initial=torch.tensor(base[:,:3].copy(),device='cuda');logs0=torch.tensor(base[:,7:10].copy(),device='cuda')
    model={k:torch.nn.Parameter(torch.tensor(x.copy(),device='cuda')) for k,x in dict(means=base[:,:3],logs=base[:,7:10],quats=base[:,10:14],opacity=base[:,6]).items()}
    dc=torch.nn.Parameter(torch.tensor(sh[:,:1].copy(),device='cuda'));rest=torch.nn.Parameter(torch.tensor(sh[:,1:].copy(),device='cuda'))
    optimizer=torch.optim.Adam([dict(params=[dc],lr=.003),dict(params=[rest],lr=.001),dict(params=[model['means']],lr=.00015),
        dict(params=[model['logs']],lr=.0006),dict(params=[model['quats']],lr=.0003),dict(params=[model['opacity']],lr=.003)],eps=1e-8)
    teacher=load_scene(a.model);reference={}
    with torch.no_grad():
        for i in ids:
            im,aa,_=render(teacher,views[i],a.size,True);reference[i]=(im[:,:,3].detach(),aa.detach())
    del teacher;torch.cuda.empty_cache()
    order=[];history=[];start=time.monotonic()
    for step in range(a.steps):
        if not order:order=list(rng.permutation(ids))
        i=int(order.pop());truth=torch.tensor(target[i].copy(),device='cuda',dtype=torch.float32)/255
        optimizer.zero_grad(set_to_none=True);model['sh']=torch.cat([dc,rest],1)
        im,alpha,normals,surface,dist,median,_=render_surface(model,views[i],a.size)
        rgb=im[0,:,:,:3];aa=alpha[0,:,:,0];td,ta=reference[i]
        valid=(ta>.98)&(td>.1)&(td<30);warm=min(1.,max(0.,(step-150)/500))
        photo=.8*(rgb-truth).abs().mean()+.2*structure_loss(rgb,truth)
        n=F.normalize(normals[0],dim=-1);s=F.normalize(surface,dim=-1)
        normal=(1-(n*s).sum(-1)).clamp(0,2)
        stable=valid&(aa.detach()>.95)&(s.detach().square().sum(-1)>.5)
        nloss=normal[stable].mean() if stable.any() else normal.mean()*0
        dloss=dist.mean()
        loss=photo+warm*(a.normal_weight*nloss+a.distortion_weight*dloss)+.04*(aa-ta).abs().mean()+.0002*rest.square().mean()
        if valid.any():loss+=.01*((median[0,:,:,0]-td).abs()/td.clamp(min=.5)).clamp(max=1)[valid].mean()
        loss+=.001*(model['means']-initial).square().mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite surface loss')
        loss.backward()
        if step<100:
            for k in ['means','logs','quats']:model[k].grad=None
        optimizer.step()
        with torch.no_grad():
            model['means'].copy_(initial+(model['means']-initial).clamp(-.12,.12))
            model['logs'].copy_(torch.maximum(torch.minimum(model['logs'],logs0+.35),logs0-.5))
            model['quats'].copy_(F.normalize(model['quats'],dim=1));model['opacity'].clamp_(-8,10);dc.clamp_(-.5/SH0,.5/SH0);rest.clamp_(-2,2)
        if step%100==0 or step==a.steps-1:
            rec=dict(step=step+1,loss=float(loss.detach()),photo=float(photo.detach()),normal=float(nloss.detach()),distortion=float(dloss.detach()),seconds=time.monotonic()-start,gpu_gb=torch.cuda.max_memory_allocated()/1e9)
            history.append(rec);print(rec,flush=True)
        if (step+1)%500==0 or step==a.steps-1:
            coeff=torch.cat([dc,rest],1).detach().cpu().numpy()
            b=np.c_[model['means'].detach().cpu().numpy(),coeff[:,0],model['opacity'].detach().cpu().numpy(),model['logs'].detach().cpu().numpy(),model['quats'].detach().cpu().numpy()].astype('f')
            np.savez(a.out/'optimized.npz',base=b,sh=coeff)
            write_json(a.out/'training.json',dict(representation='2D surface Gaussians',not_standard_3dgs_ply=True,source_views=len(ids),steps=step+1,history=history,
                config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},reference_model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest()))
    with torch.no_grad():
        for i in ids:
            im,aa,*_=render_surface(model,views[i],a.size)
            cv2.imwrite(str(a.out/(views[i]['name']+'.jpg')),np.uint8(np.clip(im[0,:,:,:3].cpu().numpy()[:,:,::-1],0,1)*255))


def capture(a):
    import torch,cv2
    from refit_render import load_scene
    torch.set_num_threads(4);model=load_scene(a.model);meta=load(a.source)
    cameras=np.load(a.source/'aligned-cameras.npz');C,Q=cameras['C'],cameras['Q']
    a.out.mkdir(parents=True,exist_ok=True)
    for name in ['scene.json','aligned-cameras.npz']:shutil.copyfile(a.source/name,a.out/name)
    (a.out/'data-512').mkdir(exist_ok=True);shutil.copyfile(a.source/'data-512/dataset.json',a.out/'data-512/dataset.json')
    dataset=json.loads((a.source/'data-512/dataset.json').read_text('utf8'))
    positions=[(s['id'],C[i],Q[i],True) for i,s in enumerate(meta['stations'])]
    for i,j in dataset['midpoint_edges']:positions.append((f'mid-{i}-{j}',(C[i]+C[j])/2,Q[i],False))
    frames=[]
    with torch.no_grad():
        for label,c,q,original in positions:
            folder=a.out/('warped-depth' if original else 'virtual-depth')/label;folder.mkdir(parents=True,exist_ok=True)
            for k,b in enumerate(prep.FACES):
                v=dict(viewmat=viewmat(c,q@b).tolist());im,aa,n,s,dist,med,_=render_surface(model,v,a.size)
                rgb=im[0,:,:,:3].cpu().numpy();alpha=aa[0,:,:,0].cpu().numpy();depth=med[0,:,:,0].cpu().numpy()
                np.savez_compressed(folder/f'depth-{k}.npz',depth=depth,alpha=alpha)
                cv2.imwrite(str(folder/f'color-{k}.jpg'),np.uint8(np.clip(rgb[:,:,::-1],0,1)*255))
                frames.append(dict(depth=str((folder/f'depth-{k}.npz').relative_to(a.out)),color=str((folder/f'color-{k}.jpg').relative_to(a.out)),viewmat=v['viewmat'],original_camera=original))
            print('Captured median surface depth',label,flush=True)
    bounds=np.quantile(model['means'].cpu().numpy(),[.005,.995],axis=0)
    write_json(a.out/'capture.json',dict(size=a.size,frames=frames,bounds=bounds.tolist(),model=str(a.model),model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),synthetic_depth=True,depth_mode='2DGS median',representation='2D surface Gaussians'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','capture']);p.add_argument('--source',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--size',type=int,default=512);p.add_argument('--steps',type=int,default=2500);p.add_argument('--resume',type=Path)
    p.add_argument('--normal-weight',type=float,default=.05);p.add_argument('--distortion-weight',type=float,default=.01)
    a=p.parse_args();globals()[a.stage](a)
