"""Reproduce captured browser cameras in gsplat and compare with source photo."""
import argparse,json
from pathlib import Path
import numpy as np,cv2 as cv,torch
from gsplat import rasterization,spherical_harmonics
from core import load,write_json
from refit_render import load_scene
from photo_recolor import sample_erp

def main(a):
    meta=load(a.scene);cams=np.load(a.scene/'aligned-cameras.npz');model=load_scene(a.model)
    photo=cv.imdecode(np.fromfile(meta['stations'][0]['photo'],np.uint8),1)[:,:,::-1]
    records=[]
    for face in [4,5]:
        m=json.loads((a.out/f'face-{face}.json').read_text());w,h=m['width'],m['height']
        v=np.diag([1,-1,-1,1])@np.array(m['worldInverse']).reshape(4,4).T
        p=np.array(m['projection']).reshape(4,4).T;K=np.array([[p[0,0]*w/2,0,w/2],[0,p[1,1]*h/2,h/2],[0,0,1]])
        center=np.array(m['position']);y,x=np.indices((h,w));rays=np.stack([(x+.5-w/2)/K[0,0],(y+.5-h/2)/K[1,1],np.ones_like(x)],-1)
        points=rays.reshape(-1,3)@v[:3,:3]+center
        target=sample_erp(photo,points,cams['C'][0],cams['Q'][0]).reshape(h,w,3)/255.
        colors=(.5+spherical_harmonics(2,model['means']-torch.tensor(center,dtype=torch.float32,device='cuda'),model['sh'])).clamp(0,1)
        with torch.no_grad():rgb,alpha,_=rasterization(model['means'],model['quats'],model['logs'].exp(),model['opacity'].sigmoid(),colors.float(),torch.tensor(v[None],dtype=torch.float32,device='cuda'),torch.tensor(K[None],dtype=torch.float32,device='cuda'),w,h,near_plane=.1,far_plane=100,packed=True,eps2d=.3)
        rgb=rgb[0].cpu().numpy();browser=cv.imread(str(a.out/f'face-{face}.png'))[:,:,::-1]/255.
        cv.imwrite(str(a.out/f'face-{face}-gpu.png'),(rgb.clip(0,1)[:,:,::-1]*255).astype('u1'))
        cv.imwrite(str(a.out/f'face-{face}-photo.png'),(target[:,:,::-1]*255).astype('u1'))
        row=dict(face=face,browser_gpu_mae=float(abs(browser-rgb).mean()),browser_photo_mae=float(abs(browser-target).mean()),gpu_photo_mae=float(abs(rgb-target).mean()))
        records.append(row);print(row,flush=True)
    write_json(a.out/'parity.json',records)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
