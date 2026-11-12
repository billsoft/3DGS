"""Fail early on CUDA/projection/gradient problems before fitting private data."""
from pathlib import Path
import argparse, json
import numpy as np
from PIL import Image
import torch
from refit_render import load_scene, render, load_dataset

parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
args=parser.parse_args();meta,targets=load_dataset(args.data)
scene=load_scene(args.data/'native-2.npy')
scene['colors'].requires_grad_()
im,alpha,_=render(scene,meta['views'][4],meta['size'])
loss=(im-torch.tensor(targets[4],device='cuda')/255).abs().mean();loss.backward()
assert scene['colors'].grad is not None and torch.isfinite(scene['colors'].grad).all()
assert scene['colors'].grad.abs().sum()>0
Image.fromarray((im.detach().clamp(0,1).cpu().numpy()*255).astype('u1')).save(args.data/'cuda-probe.png')
print(json.dumps(dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),
    photo_mae=float(loss),gradient_finite=True,alpha_mean=float(alpha.mean()))),flush=True)
