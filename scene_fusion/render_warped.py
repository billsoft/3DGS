"""Refresh visibility/depth after deformation; never use stale pre-warp depths."""
import argparse
from pathlib import Path
import numpy as np
import torch
from core import load,prep,viewmat
from refit_render import load_scene,render

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();meta=load(a.out);cam=np.load(a.out/'aligned-cameras.npz')
for i,item in enumerate(meta['stations']):
    folder=a.out/'warped-depth'/item['id'];folder.mkdir(parents=True,exist_ok=True)
    C,Q=cam['C'][i],cam['Q'][i];np.savez(folder/'cache.npz',R=Q.T,C=C)
    scene=load_scene(a.out/'warped'/(item['id']+'.npy'))
    with torch.no_grad():
        for k,b in enumerate(prep.FACES):
            basis=Q@b;im,alpha,_=render(scene,dict(viewmat=viewmat(C,basis).tolist()),640,depth=True)
            np.savez_compressed(folder/f'depth-{k}.npz',depth=im[:,:,3].cpu().numpy(),alpha=alpha.cpu().numpy(),basis=basis)
    del scene;torch.cuda.empty_cache();print('Warped depth',item['id'],flush=True)
