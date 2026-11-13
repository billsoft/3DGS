"""Higher-resolution static-PLY distillation with error-driven tangent splitting.

Uses a deterministic spatial teacher, not the player's history-dependent rest
switch. Synthetic supervision is never described as observed scene geometry.
"""
import argparse,json,time,os,hashlib,shutil
from pathlib import Path
import numpy as np
from core import write_json
from tour_distill import tour_weights

def at_original_station(view,centers):
    return bool(np.min(np.linalg.norm(np.asarray(centers)-view['center'],axis=1))<1e-6)


def split_arrays(base,sh,ids):
    from scipy.spatial.transform import Rotation
    ids=np.asarray(ids,dtype=int);keep=np.ones(len(base),bool);keep[ids]=False
    child=np.repeat(base[ids],2,axis=0);coeff=np.repeat(sh[ids],2,axis=0)
    axis=np.argmax(child[:,7:10],axis=1);rot=Rotation.from_quat(child[:,[11,12,13,10]]).as_matrix()
    sigma=np.exp(child[np.arange(len(child)),7+axis]);direction=rot[np.arange(len(child)),:,axis]
    child[:,:3]+=direction*(sigma*np.tile([-.4,.4],len(ids)))[:,None]
    child[np.arange(len(child)),7+axis]+=np.log(.75)
    alpha=1/(1+np.exp(-child[:,6]));alpha=np.clip(alpha*.7,1e-6,1-1e-6);child[:,6]=np.log(alpha/(1-alpha))
    return np.concatenate([base[keep],child]),np.concatenate([sh[keep],coeff])


def run(a):
    import torch
    import torch.nn.functional as F
    from refit_render import load_scene,render,SH0
    from refit_train import structure_loss
    if a.out.resolve()==a.reference.resolve() or (a.out/'optimized.npz').exists():
        raise ValueError('Use a fresh candidate directory; never overwrite a reference or existing checkpoint')
    torch.set_num_threads(4);torch.manual_seed(129);rng=np.random.default_rng(129)
    a.out.mkdir(parents=True,exist_ok=True);data=a.out/'data';data.mkdir(exist_ok=True)
    meta=json.loads((a.reference/'data/dataset.json').read_text('utf8'));meta['core']=a.core;meta['comparison_model']=str(a.reference/'optimized.npz');meta['live_teacher_required']=True
    if a.views:
        extra=json.loads(a.views.read_text('utf8'))['views']
        for i,v in enumerate(extra):v.update(role='train' if i%2==0 else 'test',kind='oblique')
        meta['views']+=extra
    for v in meta['views']:v['weights']=tour_weights(np.array(v['center']),np.array(meta['centers']),a.core).tolist()
    # Link only immutable teacher assets, never the baseline checkpoint.
    for file in (a.reference/'data').glob('*.npy'):
        if file.name!='initial.npy' and not (file.stem.startswith('teacher-') and file.stem[8:].isdigit()):continue
        dest=data/file.name
        if not dest.exists():
            try:os.link(file,dest)
            except OSError:shutil.copy2(file,dest)
    write_json(data/'dataset.json',meta)
    from teacher_cache import TeacherCache
    teachers=TeacherCache(data,len(meta['centers']))
    z=np.load(a.resume or a.reference/'optimized.npz');base,coeff=z['base'],z['sh'];model={}
    def setup(base,coeff):
        m={k:torch.nn.Parameter(torch.tensor(v.copy(),device='cuda')) for k,v in dict(means=base[:,:3],logs=base[:,7:10],quats=base[:,10:14],opacity=base[:,6],dc=coeff[:,:1],rest=coeff[:,1:]).items()}
        rates=dict(means=.00005,logs=.0004,quats=.0001,opacity=.003,dc=.003,rest=.001)
        if a.appearance_only:
            for k in ('means','logs','quats'):m[k].requires_grad_(False);rates.pop(k)
        opt=torch.optim.Adam([dict(params=[m[k]],lr=lr*a.lr_scale,base_lr=lr*a.lr_scale) for k,lr in rates.items()],eps=1e-8)
        return m,opt,torch.tensor(base[:,:3].copy(),device='cuda'),torch.tensor(base[:,7:10].copy(),device='cuda')
    model,optim,initial,logs0=setup(base,coeff)
    def arrays():
        sh=torch.cat([model['dc'],model['rest']],1).detach().cpu().numpy()
        b=np.c_[model['means'].detach().cpu().numpy(),sh[:,0],model['opacity'].detach().cpu().numpy(),model['logs'].detach().cpu().numpy(),model['quats'].detach().cpu().numpy()].astype('f')
        return b,sh
    score=torch.zeros(len(base),device='cuda');count=score.clone();radius=score.clone()
    ids=[i for i,v in enumerate(meta['views']) if v['role']=='train'];anchors=[i for i in ids if meta['views'][i]['kind']=='anchor']
    original={i for i in ids if at_original_station(meta['views'][i],meta['centers'])}
    if a.anchor_probability is not None:anchors=sorted(original)
    if not anchors:raise ValueError('No original station training views')
    sampled_original=0
    order=[];history=[];splits=[];start=time.monotonic();size=a.size
    for step in range(a.steps):
        for group in optim.param_groups:group['lr']=group['base_lr']*a.final_lr_ratio**(step/max(1,a.steps-1))
        size=a.sizes[step%len(a.sizes)] if a.sizes else a.size
        if not order:order=list(rng.permutation(ids))
        anchor_step=step%3==0 if a.anchor_probability is None else rng.random()<a.anchor_probability
        j=int(rng.choice(anchors)) if anchor_step else int(order.pop());v=meta['views'][j]
        sampled_original+=int(j in original)
        with torch.no_grad():
            truth=torch.zeros((size,size,3),device='cuda');ta=torch.zeros((size,size),device='cuda');td=ta.clone()
            for i,w in enumerate(v['weights']):
                if w>0:
                    im,aa,_=render(teachers[i],v,size,True);truth+=w*im[:,:,:3];ta+=w*aa
                    if w>.999:td=im[:,:,3]
        optim.zero_grad(set_to_none=True);model['sh']=torch.cat([model['dc'],model['rest']],1);model['colors']=(model['dc'][:,0]*SH0+.5).clamp(0,1)
        im,aa,info=render(model,v,size,True);rgb=im[:,:,:3]
        if not a.appearance_only:info['means2d'].retain_grad()
        loss=.8*(rgb-truth).abs().mean()+.2*structure_loss(rgb,truth)+.08*(aa-ta).abs().mean()+.0002*model['rest'].square().mean()
        loss+=.12*((rgb[:,1:]-rgb[:,:-1]-truth[:,1:]+truth[:,:-1]).abs().mean()+(rgb[1:]-rgb[:-1]-truth[1:]+truth[:-1]).abs().mean())
        if a.patch_weight:
            # Give difficult regions their own budget instead of letting large
            # uniform surfaces dominate the whole-frame average. All channels
            # are supervised; a grayscale score cannot excuse a colored halo.
            error=rgb-truth
            chroma=error-error.mean(-1,keepdim=True)
            local=(error.abs().mean(-1)+chroma.abs().mean(-1))[None,None]
            patches=F.avg_pool2d(local,32,32).flatten()
            loss+=a.patch_weight*patches.topk(max(1,len(patches)//4)).values.mean()
        valid=(td>.1)&(td<30)&(ta>.98)
        if valid.any():loss+=.008*((im[:,:,3]-td).abs()/td.clamp(min=.5)).clamp(max=1)[valid].mean()
        if j in original:loss*=a.anchor_weight
        if not torch.isfinite(loss):raise ValueError('Nonfinite refined PLY loss')
        loss.backward()
        with torch.no_grad():
            if not a.appearance_only:
                ii=info['gaussian_ids'];grad=info['means2d'].grad.norm(dim=-1);rr=info['radii'].float()
                if rr.ndim>1:rr=rr.max(dim=-1).values
                score.index_add_(0,ii,grad);count.index_add_(0,ii,torch.ones_like(grad));radius.scatter_reduce_(0,ii,rr,reduce='amax',include_self=True)
        optim.step()
        with torch.no_grad():
            model['means'].copy_(initial+(model['means']-initial).clamp(-.06,.06));model['logs'].copy_(torch.maximum(torch.minimum(model['logs'],logs0+.35),logs0-.5))
            if not a.appearance_only:model['quats'].copy_(F.normalize(model['quats'],dim=1))
            model['opacity'].clamp_(-8,10);model['dc'].clamp_(-.5/SH0,.5/SH0);model['rest'].clamp_(-2,2)
        if not a.no_split and step+1 in a.split_steps and len(initial)<a.max_points:
            quality=score/count.clamp(min=1)*radius.sqrt();quality[(count<5)|(radius<3)]=0
            n=min(a.split_count,a.max_points-len(initial),int((quality>0).sum()))
            if n:
                chosen=torch.topk(quality,n).indices.cpu().numpy();b,s=arrays();b,s=split_arrays(b,s,chosen)
                del model,optim;torch.cuda.empty_cache();model,optim,initial,logs0=setup(b,s)
                score=torch.zeros(len(b),device='cuda');count=score.clone();radius=score.clone()
                splits.append(dict(step=step+1,parents=n,points=len(b)));print('Split',splits[-1],flush=True)
        if step%100==0 or step==a.steps-1:
            rec=dict(step=step+1,loss=float(loss.detach()),points=len(initial),seconds=time.monotonic()-start,gpu_gb=torch.cuda.max_memory_allocated()/1e9);history.append(rec);print(rec,flush=True)
        if (step+1)%500==0 or step==a.steps-1:
            b,s=arrays();np.savez(a.out/'optimized.npz',base=b,sh=s)
            write_json(a.out/'training.json',dict(steps=step+1,size=size,training_sizes=a.sizes or [a.size],points=len(b),core=a.core,history=history,splits=splits,appearance_only=a.appearance_only,synthetic_supervision=True,teacher_temporal_state=False,
                lr_scale=a.lr_scale,final_lr_ratio=a.final_lr_ratio,patch_weight=a.patch_weight,split_steps=a.split_steps,
                anchor_probability=a.anchor_probability,anchor_weight=a.anchor_weight,sampled_original_views=sampled_original,
                reference_model_sha256=hashlib.sha256((a.reference/'optimized.npz').read_bytes()).hexdigest()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--views',type=Path)
    p.add_argument('--core',type=float,default=.49);p.add_argument('--size',type=int,default=1024);p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--max-points',type=int,default=1400000);p.add_argument('--split-count',type=int,default=150000)
    p.add_argument('--resume',type=Path);p.add_argument('--no-split',action='store_true');p.add_argument('--sizes',type=int,nargs='+')
    p.add_argument('--appearance-only',action='store_true',help='Freeze positions, sizes and rotations; fit color and opacity only')
    p.add_argument('--lr-scale',type=float,default=1.);p.add_argument('--final-lr-ratio',type=float,default=1.)
    p.add_argument('--patch-weight',type=float,default=0.);p.add_argument('--split-steps',type=int,nargs='+',default=[500,1300,2100])
    p.add_argument('--anchor-probability',type=float,help='Optional probability of drawing directly from original station views, including narrow FOVs')
    p.add_argument('--anchor-weight',type=float,default=1.,help='Extra loss weight for original station views, not virtual near-station views')
    a=p.parse_args()
    if not 0<a.lr_scale or not 0<a.final_lr_ratio<=1 or a.patch_weight<0:p.error('Invalid refinement rates/weight')
    if a.anchor_weight<=0 or (a.anchor_probability is not None and not 0<=a.anchor_probability<=1):p.error('Invalid original station weighting')
    if a.appearance_only:a.no_split=True
    run(a)
