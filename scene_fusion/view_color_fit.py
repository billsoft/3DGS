"""Fit SH appearance to every source direction with fixed Gaussian geometry."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from core import write_json
from refit_render import load_scene,load_dataset,render,SH0
from refit_train import structure_loss

def surface_graph(base):
    from scipy.spatial import cKDTree
    from gaussian_geometry import shortest_axis_normals
    xyz=base[:,:3];norm=shortest_axis_normals(base);rgb=np.clip(.5+SH0*base[:,3:6],0,1)
    radius=max(float(np.percentile(np.linalg.norm(xyz-np.median(xyz,axis=0),axis=1),75)),1e-6)
    spacing=radius*.005;tree=cKDTree(xyz);parts=[]
    for start in range(0,len(base),100000):
        ids=np.arange(start,min(start+100000,len(base)));d,ix=tree.query(xyz[ids],k=min(7,len(base)),workers=4)
        src=np.repeat(ids,ix.shape[1]-1);dst=ix[:,1:].ravel();dist=d[:,1:].ravel()
        aligned=abs((norm[src]*norm[dst]).sum(1));diff=np.linalg.norm(rgb[src]-rgb[dst],axis=1)
        weight=np.exp(-(dist/spacing)**2-(diff/.08)**2)*np.clip((aligned-.8)/.2,0,1)**2
        keep=(src<dst)&(weight>.05)&(dist<spacing*1.6)
        parts.append((src[keep],dst[keep],weight[keep]))
    return dict(zip(['src','dst','weight'],[np.concatenate([p[k] for p in parts]) for k in range(3)]))

def read_base(path):
    if path.suffix=='.npz':
        with np.load(path) as data:return data['base']
    return np.load(path)

def run(a):
    a.out.mkdir(parents=True,exist_ok=True);torch.manual_seed(1729);np.random.seed(1729);torch.set_num_threads(4)
    meta,targets=load_dataset(a.data);size=meta['size'];views=meta['views'];ids=[i for i,v in enumerate(views) if v['role']=='train'];novel=[i for i,v in enumerate(views) if v['role']=='novel']
    model=load_scene(a.model);base=read_base(a.model)
    original=model['sh'].detach().clone() if 'sh' in model else torch.cat([(model['colors']-.5)[:,None,:]/SH0,torch.zeros((len(base),8,3),device='cuda')],1)
    if original.shape[1]!=9:raise ValueError('This viewer experiment uses degree two SH')
    dc=torch.nn.Parameter(original[:,:1].clone());rest=torch.nn.Parameter(original[:,1:].clone())
    protected=None
    if a.protected_appearance:
        mask=np.load(a.protected_appearance)
        if mask.dtype!=bool or mask.shape!=(len(base),) or not a.reference:raise ValueError('Protected appearance requires a boolean point mask and an identical-geometry reference')
        protected=torch.as_tensor(mask,device='cuda');accepted=read_base(a.reference)
        if accepted.shape!=base.shape or not np.array_equal(accepted[:,:3],base[:,:3]):raise ValueError('Protected reference geometry differs')
        accepted_dc=torch.as_tensor(accepted[:,3:6],device='cuda')
        with torch.no_grad():dc[protected,0]=accepted_dc[protected];rest[protected]=0
    truth=torch.tensor(targets,device='cuda',dtype=torch.float32)/255
    graph=None;reference_model=load_scene(a.reference) if a.reference else model
    if a.graph_weight:
        anchor=read_base(a.reference) if a.reference else base
        if anchor.shape!=base.shape or not np.array_equal(anchor[:,:3],base[:,:3]):raise ValueError('Graph reference must use identical geometry and point ordering')
        graph={k:torch.as_tensor(v,device='cuda') for k,v in surface_graph(anchor).items()}
        anchor_dc=torch.as_tensor(anchor[:,3:6],device='cuda')
        if not len(graph['src']):raise ValueError('Surface graph has no valid edges')
        print('Surface graph edges',len(graph['src']),flush=True)
    reference={}
    with torch.no_grad():
        for i,v in enumerate(views):
            if v['role']=='novel':
                im,_,_=render(reference_model,v,size);reference[i]=im.detach()
    optim=torch.optim.Adam([dict(params=[dc],lr=a.dc_lr),dict(params=[rest],lr=a.sh_lr)],eps=1e-8)
    records=[];order=[];started=time.monotonic()
    for step in range(a.steps):
        if not order:order=list(np.random.permutation(ids))
        is_novel=step%10==9 and bool(novel);i=novel[(step//10)%len(novel)] if is_novel else order.pop()
        optim.zero_grad(set_to_none=True);model['sh']=torch.cat([dc,rest],1);im,alpha,_=render(model,views[i],size)
        if is_novel:loss=.06*(im-reference[i]).abs().mean()
        else:
            loss=.8*(im-truth[i]).abs().mean()+.2*structure_loss(im,truth[i])
            dx=(im[:,1:]-im[:,:-1])-(truth[i,:,1:]-truth[i,:,:-1]);dy=(im[1:]-im[:-1])-(truth[i,1:]-truth[i,:-1])
            loss=loss+.3*(dx.abs().mean()+dy.abs().mean())
            if a.blotch_weight:
                low=lambda x:F.avg_pool2d(x.permute(2,0,1)[None].contiguous(),31,16,15,count_include_pad=False)
                loss=loss+a.blotch_weight*(low(im)-low(truth[i])).abs().mean()
        loss=loss+a.regularization*rest.square().mean()
        if graph is not None:
            chosen=torch.randint(len(graph['src']),(131072,),device='cuda');src,dst=graph['src'][chosen],graph['dst'][chosen]
            change=(dc[:,0]-anchor_dc)*SH0
            dc_cost=(change[src]-change[dst]).square().mean(1)
            # Integrated directional color difference, not averaging original texture.
            sh_cost=(rest[src]-rest[dst]).square().sum(1).mean(1)/(4*np.pi)
            loss=loss+a.graph_weight*(graph['weight'][chosen]*(dc_cost+sh_cost)).mean()
        loss.backward();optim.step()
        with torch.no_grad():
            dc.clamp_(-.5/SH0,.5/SH0);rest.clamp_(-a.coefficient_limit,a.coefficient_limit)
            if protected is not None:dc[protected,0]=accepted_dc[protected];rest[protected]=0
        if step%100==0 or step==a.steps-1:
            record=dict(step=step+1,loss=float(loss.detach()),seconds=time.monotonic()-started,peak_gb=torch.cuda.max_memory_allocated()/1e9);records.append(record);print(record,flush=True)
        if (step+1)%1000==0 or step==a.steps-1:
            coeff=torch.cat([dc,rest],1).detach().cpu().numpy();b=base.copy();b[:,3:6]=coeff[:,0];np.savez(a.out/'optimized.npz',base=b,sh=coeff)
    write_json(a.out/'training.json',dict(degree=2,color_range='per-Gaussian clamp [0,1] before compositing',source_views=len(ids),geometry_frozen=True,opacity_frozen=True,config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},records=records))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=4000);p.add_argument('--dc-lr',type=float,default=.002);p.add_argument('--sh-lr',type=float,default=.004);p.add_argument('--regularization',type=float,default=.0005);p.add_argument('--coefficient-limit',type=float,default=2.)
    p.add_argument('--graph-weight',type=float,default=0.);p.add_argument('--reference',type=Path,help='Fixed color and novel-view trust reference with identical point order')
    p.add_argument('--blotch-weight',type=float,default=0.);p.add_argument('--protected-appearance',type=Path,help='Preserve already validated surface appearance; no semantic object labels')
    run(p.parse_args())
