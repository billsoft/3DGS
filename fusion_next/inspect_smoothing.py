"""Render diagnostic close-ups; object coordinates select cameras, not edits."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from refit_render import load_scene,render

p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
meta=json.loads(Path('four-room/shape-data/dataset.json').read_text())
up=np.array(meta['up']);views=[]
for name,center,target in [('globe-front',[.035,-.05,-.56],[.035,.58,-.81]),('globe-side',[-.645,.58,-.56],[.035,.58,-.81])]:
    c=np.array(center);z=np.array(target)-c;z/=np.linalg.norm(z);x=np.cross(z,up);x/=np.linalg.norm(x)
    basis=np.stack([x,np.cross(z,x),z],1);v=np.eye(4);v[:3,:3]=basis.T;v[:3,3]=-basis.T@c
    views.append(dict(name=name,position=center,target=target,viewmat=v.tolist()))
ims={};alphas={}
for label,path in [('baseline',Path('four-room/shape-free/optimized.npy')),('candidate',a.candidate)]:
    scene=load_scene(path)
    with torch.no_grad():
        for v in views:
            im,alpha,_=render(scene,v,1024);ims[label,v['name']]=im.clamp(0,1).cpu().numpy();alphas[label,v['name']]=alpha.cpu().numpy()
    del scene;torch.cuda.empty_cache()
for v in views:
    sheet=Image.new('RGB',(2048,1054));d=ImageDraw.Draw(sheet)
    for col,label in enumerate(['baseline','candidate']):
        sheet.paste(Image.fromarray((ims[label,v['name']]*255).astype('u1')),(col*1024,30));d.text((col*1024+8,8),label,fill='white')
    sheet.save(a.out/(v['name']+'.jpg'),quality=97)
    v['new_holes']=int(((alphas['baseline',v['name']]>.95)&(alphas['candidate',v['name']]<.5)).sum())
(a.out/'views.json').write_text(json.dumps(views,indent=2))
