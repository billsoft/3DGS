"""Differentiable standard-PLY rendering; no per-object or per-view appearance."""
from pathlib import Path
import json
import numpy as np
import torch
from gsplat import rasterization

SH0 = .2820947918


def load_scene(path, device='cuda'):
    a = np.load(path)
    return {k: torch.as_tensor(v.copy(), device=device) for k, v in dict(
        means=a[:, :3], colors=np.clip(a[:, 3:6]*SH0+.5, 0, 1),
        opacity=a[:, 6], logs=a[:, 7:10], quats=a[:, 10:14]).items()}


def render(scene, view, size, depth=False):
    device = scene['means'].device
    viewmat = torch.tensor(view['viewmat'], device=device, dtype=torch.float32)[None]
    K = torch.tensor([[size/2, 0, size/2], [0, size/2, size/2], [0, 0, 1]],
                     device=device, dtype=torch.float32)[None]
    rgb, alpha, info = rasterization(scene['means'], scene['quats'],
        scene['logs'].exp(), scene['opacity'].sigmoid(), scene['colors'],
        viewmat, K, size, size, near_plane=.1, far_plane=100,
        packed=True, render_mode='RGB+ED' if depth else 'RGB',
        rasterize_mode='classic', eps2d=.3)
    return rgb[0], alpha[0, ..., 0], info


def save_scene(path, scene):
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
    return json.loads((folder/'dataset.json').read_text()), np.load(folder/'targets.npy')
