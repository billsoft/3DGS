"""Bounded multi-view Gaussian refit, with coverage and novel-view trust regions.

This is an experimental optimizer, not a claim that four views determine hidden geometry.
No mesh resampling, semantic lamp patches, point deletion, or SH view memorization.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from refit_render import load_scene, render, save_scene, load_dataset


def structure_loss(a, b):
    # Explicit NCHW-contiguous storage is essential on the tested Windows
    # PyTorch 2.7 runtime: the channels-last avg-pool backward failed finite
    # differences (18x error). Float64 stabilizes flat-region variances too.
    a, b = a.double().permute(2, 0, 1)[None].contiguous(), b.double().permute(2, 0, 1)[None].contiguous()
    mean = lambda x: F.avg_pool2d(x, 7, 1, 3)
    ma, mb = mean(a), mean(b)
    va, vb = mean(a*a)-ma*ma, mean(b*b)-mb*mb
    cov = mean(a*b)-ma*mb
    ssim = ((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    return (1-ssim).mean()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--warmup', type=int, default=240)
    parser.add_argument('--mode', choices=['color', 'joint'], default='joint')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--move-limit', type=float, default=.045)
    parser.add_argument('--ssim-weight', type=float, default=.2)
    parser.add_argument('--graph-weight', type=float, default=0.)
    parser.add_argument('--edge-weight', type=float, default=.08)
    parser.add_argument('--color-lr', type=float, default=.008)
    parser.add_argument('--position-lr', type=float, default=.00016)
    parser.add_argument('--scale-lr', type=float, default=.001)
    parser.add_argument('--shape-free', action='store_true',
        help='Also fit Gaussian rotation and opacity after color warmup (joint mode only)')
    parser.add_argument('--opacity-min', type=float, default=-2.)
    parser.add_argument('--opacity-max', type=float, default=8.)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1729); np.random.seed(1729)
    torch.set_num_threads(4)
    meta, targets = load_dataset(args.data)
    size = meta['size']; views = meta['views']
    initial = load_scene(args.data/'initial.npy')
    scene = load_scene(args.resume or args.data/'initial.npy')
    train_ids = [i for i, v in enumerate(views) if v['role']=='train']
    novel_ids = [i for i, v in enumerate(views) if v['role']=='novel']
    truth = torch.tensor(targets, device='cuda', dtype=torch.float32)/255
    graph=None
    if args.graph_weight:
        graph={k:torch.tensor(v,device='cuda') for k,v in np.load(args.data/'neighbors.npz').items()}
    # Baseline references stay fixed throughout optimization, including alpha.
    reference, alphas = {}, {}
    with torch.no_grad():
        for i, v in enumerate(views):
            im, alpha, _ = render(initial, v, size)
            reference[i], alphas[i] = im.detach(), alpha.detach()
    for key in ['means', 'colors', 'logs'] + (['quats','opacity'] if args.shape_free else []):
        scene[key] = torch.nn.Parameter(scene[key])
    optim = torch.optim.Adam([
        {'params': [scene['colors']], 'lr': args.color_lr},
        {'params': [scene['means']], 'lr': args.position_lr},
        {'params': [scene['logs']], 'lr': args.scale_lr}] + ([
        {'params': [scene['quats']], 'lr': .0002},
        {'params': [scene['opacity']], 'lr': .01}] if args.shape_free else []), eps=1e-8)
    records = []; started = time.monotonic(); order = []
    for step in range(args.steps):
        if not order: order = list(np.random.permutation(train_ids))
        novel = (step % 5 == 4)
        i = novel_ids[(step//5) % len(novel_ids)] if novel else order.pop()
        optim.zero_grad(set_to_none=True)
        im, alpha, _ = render(scene, views[i], size)
        if novel:
            # Baseline is only a conservative trust region, not ground truth.
            loss = .12*(im-reference[i]).abs().mean()
        else:
            loss = (1-args.ssim_weight)*(im-truth[i]).abs().mean()
            if args.ssim_weight:
                loss = loss + args.ssim_weight*structure_loss(im, truth[i])
            dx = (im[:, 1:]-im[:, :-1])-(truth[i, :, 1:]-truth[i, :, :-1])
            dy = (im[1:]-im[:-1])-(truth[i, 1:]-truth[i, :-1])
            loss = loss + args.edge_weight*(dx.abs().mean()+dy.abs().mean())
        coverage = F.relu(torch.minimum(alphas[i], torch.tensor(.98, device='cuda'))-alpha).mean()
        loss = loss + 2*coverage
        loss = loss + .02*((scene['means']-initial['means'])/args.move_limit).square().mean()
        loss = loss + .002*(scene['logs']-initial['logs']).square().mean()
        if graph is not None:
            chosen=torch.randint(len(graph['src']),(131072,),device='cuda')
            src,dst=graph['src'][chosen].long(),graph['dst'][chosen].long()
            weights=graph['weight'][chosen]
            color_cost=(scene['colors'][src]-scene['colors'][dst]).square().mean(1)
            shift_src=scene['means'][src]-initial['means'][src]
            shift_dst=scene['means'][dst]-initial['means'][dst]
            deformation_cost=((shift_src-shift_dst)/args.move_limit).square().mean(1)
            loss=loss+args.graph_weight*(weights*(color_cost+.02*deformation_cost)).mean()
        loss.backward()
        if args.mode=='color' or step<args.warmup:
            scene['means'].grad = None; scene['logs'].grad = None
            if args.shape_free:
                scene['quats'].grad = None; scene['opacity'].grad = None
        optim.step()
        with torch.no_grad():
            scene['colors'].clamp_(0, 1)
            delta = scene['means']-initial['means']
            norm = delta.norm(dim=1, keepdim=True).clamp_min(1e-9)
            scene['means'].copy_(initial['means']+delta*(args.move_limit/norm).clamp_max(1))
            scene['logs'].copy_(torch.maximum(initial['logs']+np.log(.8),
                                torch.minimum(initial['logs']+np.log(1.25), scene['logs'])))
            if args.shape_free and args.mode!='color' and step>=args.warmup:
                scene['quats'].copy_(F.normalize(scene['quats'],dim=1))
                scene['opacity'].clamp_(args.opacity_min,args.opacity_max)
        if step%50==0 or step==args.steps-1:
            record = dict(step=step+1, loss=float(loss.detach()), coverage_loss=float(coverage.detach()),
                seconds=time.monotonic()-started, peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
            records.append(record); print(json.dumps(record), flush=True)
            (args.out/'progress.json').write_text(json.dumps(records, indent=2))
        if (step+1) % 400==0 or step==args.steps-1:
            save_scene(args.out/'optimized.npy', scene)
    with torch.no_grad():
        movement = (scene['means']-initial['means']).norm(dim=1)
        stats = dict(max_movement=float(movement.max()), mean_movement=float(movement.mean()),
            same_point_count=True,
            fixed_opacity=not (args.shape_free and args.mode=='joint' and args.steps>args.warmup),
            fixed_rotations=not (args.shape_free and args.mode=='joint' and args.steps>args.warmup),
            semantic_rules=False, view_dependent_colors=False,
            config={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, records=records)
    (args.out/'training.json').write_text(json.dumps(stats, indent=2))


if __name__=='__main__':
    run()
