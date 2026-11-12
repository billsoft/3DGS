"""Render candidates at identical source and midpoint poses; no invented truth."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scene_fusion'))
from core import prep,viewmat
from refit_render import load_scene,render

p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--models',type=Path,nargs='+',required=True);p.add_argument('--labels',nargs='+',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--size',type=int,default=768);a=p.parse_args()
assert len(a.models)==len(a.labels);a.out.mkdir(parents=True,exist_ok=True);c=np.load(a.scene/'aligned-cameras.npz');C,Q=c['C'],c['Q'];S=a.size
views=[('source-horizontal',C[0],3),('mid-first-horizontal',(C[0]+C[1])/2,3),('mid-last-horizontal',(C[0]+C[-1])/2,3),('source-up',C[0],4),('source-down',C[0],5)]
sheets={name:Image.new('RGB',(S*len(a.models),S+30)) for name,_,_ in views}
for j,(path,label) in enumerate(zip(a.models,a.labels)):
    model=load_scene(path)
    for name,center,face in views:
        with torch.no_grad():rgb,_,_=render(model,dict(viewmat=viewmat(center,Q[0]@prep.FACES[face]).tolist()),S)
        im=Image.fromarray((rgb.clamp(0,1).cpu().numpy()*255).astype('u1'));im.save(a.out/(name+'-'+str(j)+'.png'))
        sheets[name].paste(im,(j*S,30));ImageDraw.Draw(sheets[name]).text((j*S+8,8),label,fill='white')
    del model;torch.cuda.empty_cache()
for name,sheet in sheets.items():sheet.save(a.out/(name+'.jpg'),quality=96)
(a.out/'views.json').write_text(json.dumps([dict(name=n,center=x.tolist(),basis=(Q[0]@prep.FACES[k]).tolist()) for n,x,k in views],indent=2),encoding='utf8')
print('Rendered',len(views),'identical poses for',len(a.models),'candidates',flush=True)
