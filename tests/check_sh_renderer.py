"""GPU check for SH convention and legacy degree-zero rendering parity."""
import sys,json
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fusion_next'))
from refit_render import render,SH0

model=dict(means=torch.tensor([[0.,0.,3.]],device='cuda'),quats=torch.tensor([[1.,0.,0.,0.]],device='cuda'),logs=torch.tensor([[-2.,-2.,-2.]],device='cuda'),opacity=torch.tensor([6.],device='cuda'),colors=torch.tensor([[.4,.5,.6]],device='cuda'))
view=dict(viewmat=np.eye(4).tolist())
with torch.no_grad():
    rgb,alpha,_=render(model,view,64);sh=torch.zeros((1,9,3),device='cuda');sh[:,0]=(model['colors']-.5)/SH0;model['sh']=sh
    zero,az,_=render(model,view,64);assert torch.allclose(rgb,zero,atol=1e-6);assert torch.equal(alpha,az)
    # Positive camera-to-Gaussian z: first-degree z coefficient contributes +C1.
    model['sh'][0,2,0]=.2;angled,_,_=render(model,view,64)
    expected=.2*.4886025119*alpha;assert torch.allclose(angled[:,:,0]-zero[:,:,0],expected,atol=2e-6)
    model['sh'][0,2,0]=10.;saturated,_,_=render(model,view,64)
    assert torch.allclose(saturated[:,:,0],alpha,atol=2e-6),'Color must saturate before compositing, like the browser'
print(json.dumps(dict(zero_order_parity=True,direction_sign=True,alpha_unchanged=True,pre_composite_saturation=True)))
