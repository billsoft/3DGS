"""Finite-difference directional check of the actual training objective."""
from pathlib import Path
import torch, json
from refit_render import load_scene, load_dataset, render
from refit_train import structure_loss

data=Path('four-room/refit-data');meta,targets=load_dataset(data)
scene=load_scene(data/'initial.npy');scene['colors'].requires_grad_()
i=4;target=torch.tensor(targets[i],device='cuda',dtype=torch.float32)/255
def objective(mode='full'):
    im,_,_=render(scene,meta['views'][i],meta['size'])
    if mode=='l1':return (im-target).abs().mean()
    if mode=='sum':return im.mean()
    return .8*(im-target).abs().mean()+.2*structure_loss(im,target)
for mode in ['sum','l1','full']:
    scene['colors'].grad=None
    loss=objective(mode);loss.backward();g=scene['colors'].grad.clone()
    index=int(g.abs().argmax());flat=scene['colors'].reshape(-1);base=float(flat[index])
    with torch.no_grad():
        flat[index]=base+.05;plus=float(objective(mode))
        flat[index]=base-.05;minus=float(objective(mode))
        flat[index]=base
        direction=g/g.abs().max()
        original=scene['colors'].clone()
        scene['colors'].sub_(.001*direction);down=float(objective(mode))
        scene['colors'].copy_(original)
    report=dict(mode=mode,loss=float(loss),plus=plus,minus=minus,numeric=(plus-minus)/.1,
                analytic=float(g.reshape(-1)[index]),descent_loss=down)
    print(json.dumps(report),flush=True)
    assert down<float(loss),'Gradient does not descend on the same camera'
    relative=abs(report['numeric']-report['analytic'])/max(abs(report['analytic']),1e-10)
    assert relative < (.02 if mode=='sum' else .25), 'Finite-difference gradient mismatch'
