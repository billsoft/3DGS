"""Fixed-view evaluation: photos at source cameras; coverage at novel cameras.

Source-photo scores are fitting diagnostics, not held-out scene accuracy.
Novel views have no ground truth and must also pass human visual inspection.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from refit_render import load_scene, load_dataset, render


def metrics(a, b, alpha, baseline_alpha):
    from skimage.metrics import structural_similarity
    mask = baseline_alpha > .95
    gray = lambda x: (x*np.array([.299, .587, .114])).sum(-1)
    _, similarity = structural_similarity(gray(a), gray(b), data_range=1., full=True)
    return dict(mae=float(abs(a-b)[mask].mean()), ssim=float(similarity[mask].mean()),
        holes=int((mask & (alpha<.5)).sum()), mask_pixels=int(mask.sum()))


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--baseline-label',default='baseline')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    meta, targets=load_dataset(args.data);size=meta['size']
    images={};alphas={}
    for label,path in [('baseline',args.baseline or args.data/'initial.npy'),('candidate',args.candidate)]:
        scene=load_scene(path)
        with torch.no_grad():
            for i,v in enumerate(meta['views']):
                im,alpha,_=render(scene,v,size)
                images[label,i]=im.clamp(0,1).cpu().numpy();alphas[label,i]=alpha.cpu().numpy()
        del scene
        torch.cuda.empty_cache()
    records=[]
    for i,v in enumerate(meta['views']):
        before,after=images['baseline',i],images['candidate',i]
        ba,aa=alphas['baseline',i],alphas['candidate',i]
        rec=dict(view=v['name'],role=v['role'],new_holes=int(((ba>.95)&(aa<.5)).sum()),
                 changed_mae=float(abs(before-after).mean()))
        if v['source'] is not None:
            ref=targets[i]/255
            rec['baseline']=metrics(before,ref,ba,ba);rec['candidate']=metrics(after,ref,aa,ba)
            cols=[('original photo',ref),(args.baseline_label,before),('joint refit',after)]
        else: cols=[(args.baseline_label,before),('joint refit',after)]
        sheet=Image.new('RGB',(size*len(cols),size+30),(20,24,30));draw=ImageDraw.Draw(sheet)
        for col,(label,im) in enumerate(cols):
            sheet.paste(Image.fromarray((im*255).astype('u1')),(col*size,30))
            draw.text((col*size+6,8),label,fill='white')
        sheet.save(args.out/f'{v["name"]}.jpg',quality=96)
        records.append(rec)
    sourced=[r for r in records if 'baseline' in r]
    summary=dict(source_views=len(sourced),novel_views=len(records)-len(sourced),
        baseline_mean_ssim=float(np.mean([r['baseline']['ssim'] for r in sourced])),
        candidate_mean_ssim=float(np.mean([r['candidate']['ssim'] for r in sourced])),
        baseline_mean_mae=float(np.mean([r['baseline']['mae'] for r in sourced])),
        candidate_mean_mae=float(np.mean([r['candidate']['mae'] for r in sourced])),
        improved_ssim_views=sum(r['candidate']['ssim']>r['baseline']['ssim'] for r in sourced),
        worst_ssim_delta=min(r['candidate']['ssim']-r['baseline']['ssim'] for r in sourced),
        new_holes=sum(r['new_holes'] for r in records))
    (args.out/'evaluation.json').write_text(json.dumps(dict(summary=summary,views=records,
        candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        note='Source metrics reuse fitting photographs. Novel views have no independent reference.'),indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':run()
