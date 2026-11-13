"""Distill the deterministic 44%-core native tour into a standalone SH Gaussian PLY.

Teacher images are synthetic supervision, not new measurements. The motion/rest
state and hysteresis of the interactive player are deliberately excluded. This
does not claim that a discontinuous or inconsistent light field is exactly
representable by one static Gaussian model.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from core import gs,write_json,viewmat


def tour_weights(position,centers,core=.44):
    if not 0<=core<.5 or not len(centers):raise ValueError('Require 0 <= core < 0.5 and cameras')
    distance=np.linalg.norm(np.asarray(centers)-position,axis=1);order=np.argsort(distance,kind='stable')
    w=np.zeros(len(centers));w[order[0]]=1
    if len(order)>1:
        a,b=order[:2];spacing=np.linalg.norm(centers[a]-centers[b])
        if spacing>1e-8:
            x=np.clip(1-(distance[b]-distance[a])/spacing/(1-2*core),0,1)
            w[b]=.5*x*x*(3-2*x);w[a]=1-w[b]
    return w


def sample_views(meta,core):
    C=np.array(meta['centers']);views=[]
    basis=[[np.array(next(v['basis'] for v in meta['views'] if v['source']==i and v['name'].endswith('-'+str(k)))) for k in range(6)] for i in range(len(C))]
    def add(name,center,i,role,kind):
        for k,b in enumerate(basis[i]):
            views.append(dict(name=f'{name}-{k}',center=np.asarray(center).tolist(),basis=b.tolist(),viewmat=viewmat(center,b).tolist(),
                              source=i if kind=='anchor' else None,role=role,kind=kind,weights=tour_weights(center,C,core).tolist()))
    for i,c in enumerate(C):
        add(f'source-{i+1}',c,i,'train','anchor')
        nearest=np.argsort(np.linalg.norm(C-c,axis=1))[1] if len(C)>1 else i
        spacing=np.linalg.norm(C[nearest]-c)
        for sign in [-1,1]:
            add(f'near-{i+1}-{sign}',c+sign*.10*spacing*basis[i][3][:,0],i,'train','near')
            add(f'held-near-{i+1}-{sign}',c+sign*.15*spacing*basis[i][3][:,0],i,'test','near')
    for a,b in meta['midpoint_edges']:
        for role,ts in [('train',[.12,.25,.38,.44,.50,.56,.62,.75,.88]),('test',[.19,.47,.53,.81])]:
            for t in ts:add(f'{role}-edge-{a+1}-{b+1}-{t:.2f}',(1-t)*C[a]+t*C[b],a,role,'route')
    return views


def prepare(a):
    import torch
    from input_io import read_gaussians
    from refit_render import load_scene,render
    meta=json.loads((a.native/'native.json').read_text('utf8'));views=sample_views(meta,a.core);size=a.size
    a.out.mkdir(parents=True,exist_ok=True);data=a.out/'data';data.mkdir(exist_ok=True)
    if (data/'dataset.json').exists():raise ValueError('Dataset exists; use another output directory')
    shape=(len(views),size,size)
    rgb=np.lib.format.open_memmap(data/'teacher-rgb-float.npy',mode='w+',dtype='f4',shape=shape+(3,));rgb[:]=0
    alpha=np.lib.format.open_memmap(data/'teacher-alpha.npy',mode='w+',dtype='f4',shape=shape);alpha[:]=0
    depth=np.lib.format.open_memmap(data/'teacher-depth.npy',mode='w+',dtype='f4',shape=shape);depth[:]=0
    C=np.array(meta['centers']);owned=[];stats=[];torch.set_num_threads(4)
    for i,s in enumerate(meta['stations']):
        if hashlib.sha256((a.native/s['file']).read_bytes()).hexdigest()!=s['sha256']:raise ValueError('Native PLY changed since camera calibration')
        raw=read_gaussians(a.native/s['file']);M=np.array(s['world_from_raw']);array=gs.transform(raw,M)
        world_path=data/f'teacher-{i}.npy';np.save(world_path,array)
        model=load_scene(world_path)
        weights=np.array([v['weights'][i] for v in views]);indices=np.flatnonzero(weights>0)
        with torch.no_grad():
            for j in indices:
                image,opacity,_=render(model,views[j],size,depth=True);image=image.cpu().numpy();w=weights[j]
                rgb[j]+=w*image[:,:,:3];alpha[j]+=w*opacity.cpu().numpy()
                if w>.999:depth[j]=image[:,:,3]
        # Nearest-camera point ownership is ONLY an initialization. Teacher views
        # always render the full untouched source, including outside this cell.
        keep=np.empty(len(array),bool)
        for start in range(0,len(array),100000):
            xyz=array[start:start+100000,:3]
            nearest=((xyz[:,None]-C[None])**2).sum(2).argmin(1);keep[start:start+len(xyz)]=nearest==i
        owned.append(array[keep]);stats.append(dict(source=s['label'],original=len(array),initial_kept=int(keep.sum())))
        print('Teacher',i+1,'views',len(indices),'owned',int(keep.sum()),flush=True)
        del model;torch.cuda.empty_cache()
    rgb.flush();alpha.flush();depth.flush()
    targets=np.lib.format.open_memmap(data/'targets.npy',mode='w+',dtype='u1',shape=shape+(3,))
    for j in range(len(views)):targets[j]=np.uint8(np.clip(rgb[j],0,1)*255+.5)
    targets.flush();initial=np.concatenate(owned).astype('f');np.save(data/'initial.npy',initial)
    write_json(data/'dataset.json',dict(size=size,views=views,centers=meta['centers'],up=meta['up'],midpoint_edges=meta['midpoint_edges'],
        stations=[s['label'] for s in meta['stations']],core=a.core,rest_clarity=False,temporal_smoothing=False,
        teacher='Unmodified native models, transformed to common coordinates; deterministic image-level mixing',
        source_hashes=[s['sha256'] for s in meta['stations']],native_scene=a.native.name,initialization=stats,synthetic_targets=True))
    print('Prepared',len(views),'views',len(initial),'Gaussians',flush=True)


def train(a):
    import torch
    import torch.nn.functional as F
    from refit_render import load_scene,render,SH0
    from refit_train import structure_loss
    data=a.out/'data';meta=json.loads((data/'dataset.json').read_text('utf8'));views=meta['views'];size=a.train_size or meta['size']
    live=size!=meta['size'] or meta.get('live_teacher_required',False)
    target=alphas=depths=None
    if not live:
        target=np.load(data/'targets.npy',mmap_mode='r');alphas=np.load(data/'teacher-alpha.npy',mmap_mode='r');depths=np.load(data/'teacher-depth.npy',mmap_mode='r')
    torch.set_num_threads(4);torch.manual_seed(721);rng=np.random.default_rng(721)
    model=load_scene(a.resume or data/'initial.npy');reference=np.load(data/'initial.npy')
    initial=torch.tensor(reference[:,:3],device='cuda');initial_logs=torch.tensor(reference[:,7:10],device='cuda')
    teachers=[load_scene(data/f'teacher-{i}.npy') for i in range(len(meta['centers']))] if live else None
    sh=model.get('sh',torch.zeros((len(initial),9,3),device='cuda'));sh[:,0]=(model['colors']-.5)/SH0
    for k in ['means','logs','quats','opacity']:model[k]=torch.nn.Parameter(model[k])
    dc=torch.nn.Parameter(sh[:,:1].clone());rest=torch.nn.Parameter(sh[:,1:].clone())
    optimizer=torch.optim.Adam([{'params':[dc],'lr':.007},{'params':[rest],'lr':.002},
        {'params':[model['means']],'lr':.00015},{'params':[model['logs']],'lr':.001},
        {'params':[model['quats']],'lr':.0002},{'params':[model['opacity']],'lr':.008}],eps=1e-8)
    for group in optimizer.param_groups:group['lr']*=a.lr_scale
    ids=[i for i,v in enumerate(views) if v['role']=='train'];anchors=[i for i in ids if views[i]['kind']=='anchor']
    order=[];history=[];started=time.monotonic()
    for step in range(a.steps):
        if not order:order=list(rng.permutation(ids))
        j=int(rng.choice(anchors)) if step%3==0 else int(order.pop());v=views[j]
        if teachers is None:
            truth=torch.tensor(target[j].copy(),dtype=torch.float32,device='cuda')/255
            ta=torch.tensor(alphas[j].copy(),device='cuda');td=torch.tensor(depths[j].copy(),device='cuda')
        else:
            with torch.no_grad():
                truth=torch.zeros((size,size,3),device='cuda');ta=torch.zeros((size,size),device='cuda');td=torch.zeros_like(ta)
                for i,w in enumerate(v['weights']):
                    if w>0:
                        ti,aa,_=render(teachers[i],v,size,depth=True);truth+=w*ti[:,:,:3];ta+=w*aa
                        if w>.999:td=ti[:,:,3]
        optimizer.zero_grad(set_to_none=True);model['sh']=torch.cat([dc,rest],1)
        im,alpha,_=render(model,v,size,depth=True);rgb=im[:,:,:3]
        loss=.8*(rgb-truth).abs().mean()+.2*structure_loss(rgb,truth)
        loss+=.08*((rgb[:,1:]-rgb[:,:-1]-truth[:,1:]+truth[:,:-1]).abs().mean()+(rgb[1:]-rgb[:-1]-truth[1:]+truth[:-1]).abs().mean())
        loss+=.08*(alpha-ta).abs().mean()+.0002*rest.square().mean()
        # Only pure-teacher views provide depth. A blend of two depths is not
        # treated as an observed surface. Depth is weak synthetic regularization.
        valid=(td>.1)&(td<30)&(ta>.98)
        if valid.any():loss+=.008*((im[:,:,3]-td).abs()/td.clamp(min=.5)).clamp(max=1)[valid].mean()
        loss+=.0005*(model['means']-initial).square().mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite distillation loss; refusing to export')
        loss.backward()
        if step<120:
            for k in ['means','logs','quats']:model[k].grad=None
        optimizer.step()
        with torch.no_grad():
            model['means'].copy_(initial+(model['means']-initial).clamp(-.18,.18))
            model['logs'].copy_(torch.maximum(torch.minimum(model['logs'],initial_logs+.5),initial_logs-.7))
            model['opacity'].clamp_(-8,10);model['quats'].copy_(F.normalize(model['quats'],dim=1));dc.clamp_(-.5/SH0,.5/SH0);rest.clamp_(-2,2)
        if step%100==0 or step==a.steps-1:
            rec=dict(step=step+1,loss=float(loss.detach()),seconds=time.monotonic()-started,gpu_gb=torch.cuda.max_memory_allocated()/1e9);history.append(rec);print(rec,flush=True)
        if (step+1)%500==0 or step==a.steps-1:
            coeff=torch.cat([dc,rest],1).detach().cpu().numpy()
            base=np.concatenate([model['means'].detach().cpu().numpy(),coeff[:,0],model['opacity'].detach().cpu().numpy()[:,None],model['logs'].detach().cpu().numpy(),model['quats'].detach().cpu().numpy()],1).astype('f')
            np.savez(a.out/'optimized.npz',base=base,sh=coeff)
            write_json(a.out/'training.json',dict(steps=step+1,history=history,size=size,points=len(base),sh_degree=2,core=meta['core'],synthetic_supervision=True))


def evaluate(a):
    import torch,cv2 as cv
    from refit_render import load_scene,render
    from refit_evaluate import metrics
    from PIL import Image,ImageDraw
    data=a.out/'data';meta=json.loads((data/'dataset.json').read_text('utf8'))
    targets=alphas=None
    torch.set_num_threads(4)
    model=load_scene(a.out/'optimized.npz');initial=load_scene(Path(meta['comparison_model']) if meta.get('comparison_model') else data/'initial.npy');records=[];out=a.out/a.evaluation_folder;out.mkdir(exist_ok=True)
    size=a.eval_size or meta['size'];teachers=None
    if size!=meta['size'] or meta.get('live_teacher_required',False):teachers=[load_scene(data/f'teacher-{i}.npy') for i in range(len(meta['centers']))]
    else:targets=np.load(data/'targets.npy',mmap_mode='r');alphas=np.load(data/'teacher-alpha.npy',mmap_mode='r')
    with torch.no_grad():
        for j,v in enumerate(meta['views']):
            if v['kind']!='anchor' and v['role']!='test':continue
            if teachers is None:truth=targets[j]/255.;teacher_alpha=alphas[j]
            else:
                teacher_image=torch.zeros((size,size,3),device='cuda');teacher_a=torch.zeros((size,size),device='cuda')
                for i,w in enumerate(v['weights']):
                    if w>0:
                        ti,ta,_=render(teachers[i],v,size);teacher_image+=w*ti;teacher_a+=w*ta
                truth=teacher_image.clamp(0,1).cpu().numpy();teacher_alpha=teacher_a.cpu().numpy()
            im,alpha,_=render(model,v,size);before,ba,_=render(initial,v,size)
            im=im.cpu().numpy().clip(0,1);alpha=alpha.cpu().numpy();before=before.cpu().numpy().clip(0,1)
            records.append(dict(view=v['name'],kind=v['kind'],role=v['role'],initial=metrics(before,truth,ba.cpu().numpy(),teacher_alpha),student=metrics(im,truth,alpha,teacher_alpha)))
            if a.local_checks:
                from local_quality import local_metrics
                records[-1]['local']=dict(initial=local_metrics(before,truth,teacher_alpha),student=local_metrics(im,truth,teacher_alpha))
            if len(records)%30==0:print('Evaluated views',len(records),'size',size,flush=True)
            if v['kind'] in ['anchor','oblique'] or v['name'].endswith(('-3','-4','-5')):
                sheet=Image.new('RGB',(size*3,size+26));draw=ImageDraw.Draw(sheet)
                for col,(label,img) in enumerate([(f'{meta["core"]:.0%} spatial teacher',truth),('previous PLY' if meta.get('comparison_model') else 'static initialization',before),('single PLY student',im)]):
                    sheet.paste(Image.fromarray(np.uint8(img*255)),(col*size,26));draw.text((col*size+4,6),label,fill='white')
                sheet.save(out/(v['name']+'.jpg'),quality=95)
    summary={}
    for name,subset in [('anchors',[r for r in records if r['kind']=='anchor']),('heldout',[r for r in records if r['role']=='test'])]:
        summary[name]={label:{key:float(np.mean([r[label][key] for r in subset])) for key in ['mae','ssim','holes']} for label in ['initial','student']}
        summary[name]['views']=len(subset)
    # Fidelity to teacher is necessary, but not proof of correctness or visual acceptance.
    gate=all(r['student']['ssim']>=.97 and r['student']['mae']<=.01 and r['student']['holes']==0 for r in records if r['kind']=='anchor')
    if a.local_checks:
        def local_mean(label,key):
            values=[r['local'][label][key] for r in records if r['local'][label][key] is not None]
            return float(np.mean(values)) if values else None
        local_summary={label:{key:local_mean(label,key) for key in ('chroma_mae','patch_p95','patch_max')} for label in ('initial','student')}
        regressions=sorted([dict(view=r['view'],before=r['local']['initial'],after=r['local']['student']) for r in records if r['local']['student']['patch_p95'] is not None and r['local']['initial']['patch_p95'] is not None],key=lambda r:r['after']['patch_p95']-r['before']['patch_p95'],reverse=True)
        write_json(out/'local-quality.json',dict(summary=local_summary,worst_regressions=regressions[:20],note='Local diagnostics, not a guarantee that every pixel or unseen view is preserved.'))
    write_json(out/'evaluation.json',dict(summary=summary,anchor_fidelity_passed=gate,views=records,synthetic_reference=True,size=size,comparison_model=meta.get('comparison_model'),core=meta['core'],
        model_sha256=hashlib.sha256((a.out/'optimized.npz').read_bytes()).hexdigest()))
    print(json.dumps(summary),flush=True);print('Anchor fidelity gate',gate,flush=True)


def export(a):
    from sh_io import write_sh_ply,read_sh_ply
    z=np.load(a.out/'optimized.npz');base,sh=z['base'],z['sh'];final=a.out/'final';final.mkdir(exist_ok=True)
    write_sh_ply(final/'fused.ply',base,sh);b,s=read_sh_ply(final/'fused.ply')
    if not np.array_equal(b,base) or not np.array_equal(s,sh):raise ValueError('Export roundtrip mismatch')
    meta=json.loads((a.out/'data/dataset.json').read_text('utf8'));ev=json.loads((a.out/a.evaluation_folder/'evaluation.json').read_text('utf8'))
    if ev['model_sha256']!=hashlib.sha256((a.out/'optimized.npz').read_bytes()).hexdigest():raise ValueError('Stale evaluation')
    evaluations=[]
    for folder in sorted(a.out.glob('evaluation*')):
        if (folder/'evaluation.json').is_file():
            candidate=json.loads((folder/'evaluation.json').read_text('utf8'))
            if candidate.get('model_sha256')==ev['model_sha256']:evaluations.append(dict(folder=folder.name,size=candidate['size'],anchor_fidelity_passed=candidate['anchor_fidelity_passed']))
    (a.out/'data-512').mkdir(exist_ok=True);write_json(a.out/'data-512/dataset.json',meta)
    write_json(a.out/'fusion.json',dict(optimized=True,model_folder='final',sh_degree=2,splat_alpha_threshold=0,stations=meta['stations'],centers=meta['centers'],up=meta['up'],
        native_scene=a.native.name if a.native else meta.get('native_scene'),tour_core=meta['core'],comparison_scene=Path(meta['comparison_model']).parent.name if meta.get('comparison_model') else None,report_page='tour-distill-report.html',evaluation_folder=a.evaluation_folder,evaluations=evaluations,local_quality_folder=a.evaluation_folder if (a.out/a.evaluation_folder/'local-quality.json').exists() else None,preview_title=f'{meta["core"]:.0%} 漫游蒸馏 · 单一 PLY 试验',preview_status='一个静态模型，不切换原始 PLY',
        preview_note='由原始模型漫游渲染监督训练。不是无损烘焙；机位保真门槛'+('通过' if ev['anchor_fidelity_passed'] else '尚未通过')+'，请与原始漫游对照。'))
    write_json(final/'validation.json',dict(points=len(base),degree=2,roundtrip_exact=True,independent_ply=True,teacher_models_required_at_playback=False,
        sha256=hashlib.sha256((final/'fused.ply').read_bytes()).hexdigest(),anchor_fidelity_passed=ev['anchor_fidelity_passed']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','evaluate','export']);p.add_argument('--native',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--size',type=int,default=384);p.add_argument('--core',type=float,default=.44);p.add_argument('--steps',type=int,default=1800);p.add_argument('--resume',type=Path)
    p.add_argument('--eval-size',type=int);p.add_argument('--evaluation-folder',default='evaluation')
    p.add_argument('--train-size',type=int);p.add_argument('--lr-scale',type=float,default=1.)
    p.add_argument('--local-checks',action='store_true')
    a=p.parse_args()
    if a.stage=='prepare' and a.native is None:p.error('prepare requires --native')
    globals()[a.stage](a)
