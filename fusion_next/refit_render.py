"""Differentiable Gaussian rendering, including optional direction-dependent SH."""
from pathlib import Path
import json
import numpy as np
import torch
from gsplat import rasterization,spherical_harmonics

SH0 = .2820947918


def load_scene(path, device='cuda'):
    loaded = np.load(path)
    sh=None
    if isinstance(loaded,np.lib.npyio.NpzFile):
        a=loaded['base'];sh=loaded['sh'];loaded.close()
        if sh.shape not in [(len(a),4,3),(len(a),9,3),(len(a),16,3)] or not np.isfinite(sh).all():raise ValueError('Invalid SH coefficients')
    else:a=loaded
    if a.ndim!=2 or a.shape[1]!=14 or not np.isfinite(a).all():raise ValueError('Invalid Gaussian geometry')
    scene={k: torch.as_tensor(v.copy(), device=device) for k, v in dict(
        means=a[:, :3], colors=np.clip(a[:, 3:6]*SH0+.5, 0, 1),
        opacity=a[:, 6], logs=a[:, 7:10], quats=a[:, 10:14]).items()}
    if sh is not None:scene['sh']=torch.as_tensor(sh.copy(),device=device)
    return scene


def render(scene, view, size, depth=False):
    device = scene['means'].device
    viewmat = torch.tensor(view['viewmat'], device=device, dtype=torch.float32)[None]
    focal = size/(2*np.tan(np.deg2rad(view.get('fov',90))/2))
    K = torch.tensor([[focal, 0, size/2], [0, focal, size/2], [0, 0, 1]],
                     device=device, dtype=torch.float32)[None]
    colors=scene['colors']
    if 'sh' in scene:
        degree=int(scene['sh'].shape[1]**.5)-1
        center=-(viewmat[0,:3,:3].T@viewmat[0,:3,3])
        # The browser clamps each Gaussian BEFORE alpha compositing. gsplat's
        # built-in SH path only clamps below zero; using it would train bright
        # splats that the actual viewer cannot reproduce.
        colors=(.5+spherical_harmonics(degree,scene['means']-center,scene['sh'])).clamp(0,1)
    rgb, alpha, info = rasterization(scene['means'], scene['quats'],
        scene['logs'].exp(), scene['opacity'].sigmoid(), colors,
        viewmat, K, size, size, near_plane=.1, far_plane=100,
        packed=True, render_mode='RGB+ED' if depth else 'RGB',
        rasterize_mode='classic', eps2d=.3,sh_degree=None)
    return rgb[0], alpha[0, ..., 0], info


def save_scene(path, scene):
    if 'sh' in scene:raise ValueError('Use full SH export; saving only RGB would discard directional appearance')
    array = np.concatenate([scene['means'].detach().cpu().numpy(),
        (scene['colors'].detach().cpu().numpy()-.5)/SH0,
        scene['opacity'].detach().cpu().numpy()[:, None],
        scene['logs'].detach().cpu().numpy(),
        torch.nn.functional.normalize(scene['quats'].detach(), dim=1).cpu().numpy()], 1)
    if not np.isfinite(array).all():
        raise ValueError('Non-finite optimized scene')
    np.save(path, array.astype('f'))


def load_dataset(folder):
    folder = Path(folder)
    return json.loads((folder/'dataset.json').read_text(encoding='utf8')), np.load(folder/'targets.npy')
