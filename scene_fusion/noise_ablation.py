"""Controlled color/coverage ablations. Diagnostics only, never publish as fusion."""
import argparse,json
from pathlib import Path
import numpy as np,cv2 as cv,torch
from core import load,gs,write_json
from refit_render import load_scene,render
from photo_recolor import sample_erp

def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    meta=load(a.scene);cams=np.load(a.scene/'aligned-cameras.npz')
    data=json.loads((a.scene/'data-512/dataset.json').read_text(encoding='utf8'))
    targets=np.load(a.scene/'data-512/targets.npy');source=meta['stations'][0]
    raw=gs.transform(np.load(a.scene/source['id']/'original.npy'),cams['matrices'][0])
    np.save(a.out/'native.npy',raw)
    native=load_scene(a.out/'native.npy');fused=load_scene(a.model)
    photo=cv.imdecode(np.fromfile(source['photo'],np.uint8),1)[:,:,::-1]
    projected=sample_erp(photo,raw[:,:3],cams['C'][0],cams['Q'][0])/255.
    variants={'native':native,'fused-sh':fused,
      'fused-dc':{k:v for k,v in fused.items() if k!='sh'},
      'native-photo-center':{**native,'colors':torch.as_tensor(projected,dtype=torch.float32,device='cuda')},
      'native-neutral':{**native,'colors':torch.full_like(native['colors'],.5)},
      'fused-neutral':{**{k:v for k,v in fused.items() if k!='sh'},'colors':torch.full_like(fused['colors'],.5)}}
    rows=[]
    for face in [4,5]:
        vi=next(i for i,v in enumerate(data['views']) if v['source']==0 and v['name'].endswith('-'+str(face)))
        target=targets[vi]/255.;mean=cv.GaussianBlur(target,(0,0),2)
        var=np.maximum(cv.GaussianBlur(target**2,(0,0),2)-mean**2,0).max(2)
        mask=(var<.0001)&(target.mean(2)>.12)&(target.mean(2)<.8)
        mask=cv.erode(mask.astype('u1'),np.ones((9,9),'u1'))>0
        cv.imwrite(str(a.out/f'face-{face}-photo.png'),targets[vi][:,:,::-1])
        cv.imwrite(str(a.out/f'face-{face}-flat-mask.png'),mask.astype('u1')*255)
        for name,scene in variants.items():
            with torch.no_grad():rgb,alpha,_=render(scene,data['views'][vi],512)
            rgb=rgb.cpu().numpy();alpha=alpha.cpu().numpy()
            residual=rgb-target;chroma=residual-residual.mean(2,keepdims=True)
            hp=residual-cv.GaussianBlur(residual,(0,0),2)
            rows.append(dict(face=face,variant=name,flat_pixels=int(mask.sum()),
                flat_mae=float(abs(residual[mask]).mean()),chroma_rmse=float(np.sqrt((chroma[mask]**2).mean())),
                high_frequency_rmse=float(np.sqrt((hp[mask]**2).mean())),
                rendered_high_frequency=float(np.sqrt(((rgb-cv.GaussianBlur(rgb,(0,0),2))[mask]**2).mean())),
                alpha_mean=float(alpha[mask].mean()),alpha_std=float(alpha[mask].std()),alpha_below_98=float((alpha[mask]<.98).mean())))
            cv.imwrite(str(a.out/f'face-{face}-{name}.png'),(rgb.clip(0,1)[:,:,::-1]*255).astype('u1'))
            print(face,name,rows[-1],flush=True)
    write_json(a.out/'ablation.json',dict(rows=rows,note='Same geometry per model. Native-photo-center is a single-view diagnostic, not occlusion-safe multiview recoloring. Flat mask is defined only from the original photo.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
