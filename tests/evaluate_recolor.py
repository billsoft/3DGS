"""Fixed-photo flat-region error alongside full-frame fidelity (not novel truth)."""
import argparse,sys,json,hashlib
from pathlib import Path
import numpy as np,cv2 as cv,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'fusion_next'))
from refit_render import load_scene,load_dataset,render
from refit_evaluate import metrics

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--models',type=Path,nargs='+',required=True);p.add_argument('--labels',nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
assert len(a.models)==len(a.labels);meta,targets=load_dataset(a.data);a.out.mkdir(parents=True,exist_ok=True);S=meta['size'];torch.set_num_threads(4)
references={};results=[]
for model_index,(path,label) in enumerate(zip(a.models,a.labels)):
    scene=load_scene(path);rows=[]
    for i,v in enumerate(meta['views']):
        if v['source'] is None:continue
        with torch.no_grad():rgb,alpha,_=render(scene,v,S)
        rgb=rgb.clamp(0,1).cpu().numpy();alpha=alpha.cpu().numpy();photo=targets[i].astype('f')/255
        if model_index==0:references[i]=alpha
        mean=cv.boxFilter(photo,-1,(11,11));variance=np.maximum(0,cv.boxFilter(photo*photo,-1,(11,11))-mean*mean)
        flat=(np.sqrt(variance.max(2))<.018)&(references[i]>.95)
        residual=rgb-photo;smooth=cv.GaussianBlur(residual,(0,0),2)
        row=dict(view=v['name'],**metrics(rgb,photo,alpha,references[i]),flat_pixels=int(flat.sum()),
            flat_mae=float(abs(residual)[flat].mean()),flat_highpass_error=float(abs(residual-smooth)[flat].mean()))
        rows.append(row)
    summary={key:float(np.mean([r[key] for r in rows])) for key in ['ssim','mae','flat_mae','flat_highpass_error']}
    summary['new_holes']=sum(r['holes'] for r in rows)
    record=dict(label=label,summary=summary,views=rows,sha256=hashlib.sha256(path.read_bytes()).hexdigest());results.append(record);print(label,summary,flush=True)
    del scene;torch.cuda.empty_cache()
(a.out/'evaluation.json').write_text(json.dumps(dict(size=S,results=results,note='Fixed masks from baseline alpha and original photo local variance; source fitting diagnostic only.'),indent=2),encoding='utf8')
