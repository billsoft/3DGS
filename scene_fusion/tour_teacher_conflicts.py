"""Inspect conflicting source teachers from exactly the same virtual camera."""
import argparse,json
from pathlib import Path
import numpy as np
from core import write_json

def run(a):
    import torch
    from refit_render import load_scene,render
    from PIL import Image,ImageDraw
    torch.set_num_threads(4)
    meta=json.loads((a.out/'data/dataset.json').read_text('utf8'))
    view=next(v for v in meta['views'] if v['name']==a.view)
    sources=np.argsort(np.linalg.norm(np.array(meta['centers'])-view['center'],axis=1))[:2]
    if len(sources)!=2:raise ValueError('Two source models required')
    frames=[];alphas=[]
    with torch.no_grad():
        for source in sources:
            model=load_scene(a.out/'data'/f'teacher-{source}.npy');im,alpha,_=render(model,view,a.size,True)
            frames.append(im.cpu().numpy());alphas.append(alpha.cpu().numpy());del model
        model=load_scene(a.out/'optimized.npz');student=render(model,view,a.size)[0].clamp(0,1).cpu().numpy()
    da,db=[f[:,:,3] for f in frames]
    mask=(alphas[0]>.98)&(alphas[1]>.98)&(da>.1)&(db>.1)&(da<30)&(db<30)
    delta=abs(da-db);relative=delta/np.maximum(.1,np.minimum(da,db))
    report=dict(view=a.view,sources=sources.tolist(),size=a.size,valid_pixels=int(mask.sum()),
                relative_depth_median=float(np.median(relative[mask])) if mask.any() else None,
                relative_depth_p90=float(np.percentile(relative[mask],90)) if mask.any() else None,
                fraction_above_5_percent=float((relative[mask]>.05).mean()) if mask.any() else None,
                warning='Both depths are synthetic Gaussian expected-depth renders, not measured ground truth. Same-camera disagreement identifies inconsistent supervision, not which source is correct.')
    heat=np.zeros((*mask.shape,3));heat[:,:,0]=np.clip(relative/.1,0,1)*mask
    folder=a.out/'teacher-conflict';folder.mkdir(exist_ok=True)
    sheet=Image.new('RGB',(4*a.size,a.size+25));draw=ImageDraw.Draw(sheet)
    panels=[(f'source {sources[0]+1}, same camera',frames[0][:,:,:3]),(f'source {sources[1]+1}, same camera',frames[1][:,:,:3]),('candidate single PLY',student),('relative synthetic depth difference',heat)]
    for i,(label,img) in enumerate(panels):sheet.paste(Image.fromarray(np.uint8(np.clip(img,0,1)*255)),(i*a.size,25));draw.text((i*a.size+4,5),label,fill='white')
    sheet.save(folder/(a.view+'.jpg'),quality=95);write_json(folder/(a.view+'.json'),report);print(report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--view',required=True);p.add_argument('--size',type=int,default=640);run(p.parse_args())
